import pygame
import sys
import os
import random
import numpy as np
import logging

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

os.environ["OMP_NUM_THREADS"] = "4"

# --- 1. Ορισμός του Περιβάλλοντος (Gymnasium Env) ---
class FlappyBirdEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super(FlappyBirdEnv, self).__init__()
        
        # --- Ρυθμίσεις Παραθύρου και Φυσικής ---
        self.screen_width = 864
        self.screen_height = 936
        self.speed = 4
        self.pipe_gap = 150
        self.pipe_frequency = 120 # Σε frames (περίπου 2 δευτερόλεπτα στα 60fps)
        self.gravity = 0.5
        self.jump_velocity = -10
        self.max_velocity = 8
        self.ground_level = 768
        
        # --- Ενέργειες: 0=Τίποτα, 1=Πήδα (Space) ---
        self.action_space = spaces.Discrete(2)
        
        # --- Όραση (Observation Space): 3 τιμές ---
        # 1. Ύψος Πουλιού (Y)
        # 2. Οριζόντια Απόσταση (X) μέχρι τον επόμενο σωλήνα
        # 3. Ύψος της "τρύπας" του επόμενου σωλήνα (Bottom Pipe Y)
        # Όλες οι τιμές θα κανονικοποιούνται από το 0 έως το 1.
        # ΑΛΛΑΓΗ: 4 είσοδοι (Height, X_Dist, Y_Bottom_Pipe, Velocity)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        # --- ΔΙΟΡΘΩΣΗ: Η pygame.init() και η φόρτωση assets μεταφέρθηκε στη render() ---
        # Πριν καλούνταν στον constructor, με αποτέλεσμα να σπάει σε headless περιβάλλον
        # (π.χ. κατά τη δημιουργία του env στο video callback χωρίς display).
        # Τώρα τα assets φορτώνονται μόνο όταν χρειαστεί rendering, όπως στο Snake.
        self._assets_loaded = False

    def _load_assets(self):
        """Φορτώνει τα assets μία φορά, μόνο όταν χρειαστεί rendering."""
        if self._assets_loaded:
            return
        pygame.init()
        pygame.font.init()
        self.bg_img = pygame.image.load("/home/nakos/Desktop/img/bg.png")
        self.ground_img = pygame.image.load("/home/nakos/Desktop/img/ground.png")
        self.pipe_img = pygame.image.load("/home/nakos/Desktop/img/pipe.png")
        self.bird_imgs = [pygame.image.load(f"/home/nakos/Desktop/img/bird{num}.png") for num in range(1, 4)]
        self.font = pygame.font.SysFont('Bauhaus 93', 60)
        self._assets_loaded = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Αρχικοποίηση Μεταβλητών Πουλιού
        self.bird_x = 100.0
        self.bird_y = self.screen_height / 2
        self.bird_velocity = 0.0
        self.bird_index = 0
        self.bird_counter = 0
        
        # Αρχικοποίηση Περιβάλλοντος
        self.pipes = [] # Λίστα από λεξικά {"x": int, "bottom_y": int, "passed": bool}
        self.frames_since_last_pipe = self.pipe_frequency
        self.ground_move = 0
        self.score = 0
        self.frame_iteration = 0
        
        return self._get_obs(), {}

    def _get_obs(self):
        # Βρίσκουμε τον επόμενο σωλήνα που είναι ΜΠΡΟΣΤΑ από το πουλί
        next_pipe = None
        for pipe in self.pipes:
            if pipe["x"] + 78 > self.bird_x: # Το 78 είναι περίπου το πλάτος της εικόνας του pipe
                next_pipe = pipe
                break
                
        if next_pipe is None:
            # Αν δεν υπάρχει σωλήνας στην οθόνη, δίνουμε "εικονικές" ασφαλείς τιμές
            dist_x = self.screen_width
            target_y = self.screen_height / 2
        else:
            dist_x = next_pipe["x"] - self.bird_x
            target_y = next_pipe["bottom_y"] - (self.pipe_gap / 2) # Στοχεύουμε στη μέση της τρύπας
            
        # Κανονικοποίηση (Normalization) [0, 1] για το Νευρωνικό Δίκτυο
        norm_bird_y = self.bird_y / self.screen_height
        norm_dist_x = max(0, dist_x) / self.screen_width
        norm_target_y = target_y / self.screen_height
        
        # ΑΛΛΑΓΗ: Κανονικοποίηση Ταχύτητας. Range είναι -10 έως 8 (άρα διαφορά 18).
        norm_velocity = (self.bird_velocity + 10.0) / 18.0
        
        return np.array([norm_bird_y, norm_dist_x, norm_target_y, norm_velocity], dtype=np.float32)

    def step(self, action):
        self.frame_iteration += 1
        reward = -0.1  # Αρνητικό reward
        terminated = False
        
        # --- 1. Εφαρμογή Ενέργειας (Jump) ---
        if action == 1:
            self.bird_velocity = self.jump_velocity
            
        # --- 2. Εφαρμογή Φυσικής (Gravity) ---
        self.bird_velocity += self.gravity
        if self.bird_velocity > self.max_velocity:
            self.bird_velocity = self.max_velocity
            
        self.bird_y += self.bird_velocity
        
        # --- 3. Δημιουργία και Κίνηση Σωλήνων ---
        self.frames_since_last_pipe += 1
        if self.frames_since_last_pipe > self.pipe_frequency:
            pipe_height = random.randint(-150, 150)
            bottom_y = int(self.screen_height / 2) + pipe_height + int(self.pipe_gap / 2)
            self.pipes.append({"x": self.screen_width, "bottom_y": bottom_y, "passed": False})
            self.frames_since_last_pipe = 0
            
        for pipe in self.pipes:
            pipe["x"] -= self.speed
            
            # Έλεγχος Σκορ (Αν περάσαμε τον σωλήνα)
            if pipe["x"] + 78 < self.bird_x and not pipe["passed"]:
                pipe["passed"] = True
                self.score += 1
                reward = 10.0 # Μεγάλη ανταμοιβή όταν περνάει σωλήνα
                
        # Αφαίρεση σωλήνων που βγήκαν εκτός οθόνης
        self.pipes = [p for p in self.pipes if p["x"] + 78 > 0]
        
        # --- 4. Έλεγχος Συγκρούσεων (Collisions) ---
        # A. Χτύπησε στο πάτωμα ή στο ταβάνι
        if self.bird_y >= self.ground_level or self.bird_y <= 0:
            terminated = True
            reward = -10.0
            
        # Β. Χτύπησε σε σωλήνα (Απλοποιημένο Bounding Box Collision)
        bird_rect = pygame.Rect(self.bird_x - 17, self.bird_y - 12, 34, 24) # Προσεγγιστικό μέγεθος του bird.png
        
        for pipe in self.pipes:
            top_pipe_rect = pygame.Rect(pipe["x"], 0, 78, pipe["bottom_y"] - self.pipe_gap)
            bottom_pipe_rect = pygame.Rect(pipe["x"], pipe["bottom_y"], 78, self.screen_height)
            
            if bird_rect.colliderect(top_pipe_rect) or bird_rect.colliderect(bottom_pipe_rect):
                terminated = True
                reward = -10.0

        if self.render_mode in ["human", "rgb_array"]:
            self.render()

        return self._get_obs(), reward, terminated, False, {"score": self.score}

    def render(self):
        pass # Δεν χρειάζεται για το training

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

# --- Κλάση για Αποθήκευση Μοντέλων (Χωρίς Βίντεο) ---
class MilestoneSaveCallback(BaseCallback):
    def __init__(self, save_dir, milestones, verbose=1):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.milestones = milestones
        self.recorded_milestones = set()

    def _on_step(self) -> bool:
        for m in self.milestones:
            if self.num_timesteps >= m and m not in self.recorded_milestones:
                self.recorded_milestones.add(m)
                self.model.save(os.path.join(self.save_dir, f"model_step_{m}"))
                print(f"[Milestone] Το μοντέλο αποθηκεύτηκε στο βήμα {m}")
        return True

# --- 2. Ρύθμιση Εκπαίδευσης & Callbacks ---
if __name__ == "__main__":
    EXP_NAME = "Exp_7_Velocity_Observation_and_Survival_Penalnty"
    base_dir = f"/home/nakos/Desktop/Εργασίες/Πτυχιακή/Flappy_Bird/{EXP_NAME}"
    
    log_dir = f"{base_dir}/tensorboard/"
    save_dir = f"{base_dir}/saved_models/"
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Καταγραφή των logs στο τερματικό ΚΑΙ σε αρχείο CSV
    log_file = f"{base_dir}/training_logs.txt"
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')

    def make_train_env():
        return Monitor(FlappyBirdEnv(render_mode=None))
    
    eval_env = Monitor(FlappyBirdEnv(render_mode=None))
    train_env = DummyVecEnv([make_train_env])
    
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=save_dir, 
        log_path=save_dir, 
        eval_freq=10000, 
        n_eval_episodes=5,
        deterministic=True, 
        render=False
    )
    
    milestones = [50000, 200000, 500000, 800000, 1000000]
    milestone_callback = MilestoneSaveCallback(save_dir=save_dir, milestones=milestones)
    
    callback_list = CallbackList([eval_callback, milestone_callback])

    # Προσθήκη ent_coef=0.01 και net_arch 64,64
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
    model = PPO("MlpPolicy", train_env, verbose=1, device="auto", policy_kwargs=policy_kwargs, ent_coef=0.01)
    model.set_logger(new_logger)
    
    print(f"Ξεκινάει η εκπαίδευση για το Flappy Bird ({EXP_NAME})!")
    model.learn(total_timesteps=1000000, callback=callback_list)
    
    model.save(f"{save_dir}/final_ppo_flappy_ai")
    train_env.close()
    print("Η εκπαίδευση ολοκληρώθηκε!")