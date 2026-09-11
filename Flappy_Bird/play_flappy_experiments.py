import pygame
import sys
import os
import random
import numpy as np
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces

os.environ["OMP_NUM_THREADS"] = "1"

# --- ΕΝΙΑΙΟ ΠΕΡΙΒΑΛΛΟΝ ΓΙΑ ΑΞΙΟΛΟΓΗΣΗ ---
# Δέχεται παράμετρο "obs_mode" για να προσαρμόζεται στο εκάστοτε πείραμα (3 ή 4 inputs).
class EvaluatorFlappyEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, obs_mode="basic"):
        super().__init__()
        self.obs_mode = obs_mode
        self.screen_width = 864
        self.screen_height = 936
        self.speed = 4
        self.pipe_gap = 150
        self.pipe_frequency = 120
        self.gravity = 0.5
        self.jump_velocity = -10
        self.max_velocity = 8
        self.ground_level = 768
        
        self.action_space = spaces.Discrete(2)
        
        # Ανάλογα το πείραμα, αλλάζει το μέγεθος της εισόδου
        shape_size = 4 if obs_mode == "velocity" else 3
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(shape_size,), dtype=np.float32)
        
        self.render_mode = "human"
        self.screen = None
        self.clock = None
        self._assets_loaded = False

    def _load_assets(self):
        if self._assets_loaded: return
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
        self.bird_x = 100.0
        self.bird_y = self.screen_height / 2
        self.bird_velocity = 0.0
        self.bird_index = 0
        self.bird_counter = 0
        
        self.pipes = []
        self.frames_since_last_pipe = self.pipe_frequency
        self.ground_move = 0
        self.score = 0
        
        return self._get_obs(), {}

    def _get_obs(self):
        next_pipe = None
        for pipe in self.pipes:
            if pipe["x"] + 78 > self.bird_x:
                next_pipe = pipe
                break
                
        if next_pipe is None:
            dist_x = self.screen_width
            target_y = self.screen_height / 2
        else:
            dist_x = next_pipe["x"] - self.bird_x
            target_y = next_pipe["bottom_y"] - (self.pipe_gap / 2)
            
        norm_bird_y = self.bird_y / self.screen_height
        norm_dist_x = max(0, dist_x) / self.screen_width
        norm_target_y = target_y / self.screen_height
        
        if self.obs_mode == "velocity":
            norm_velocity = (self.bird_velocity + 10.0) / 18.0
            return np.array([norm_bird_y, norm_dist_x, norm_target_y, norm_velocity], dtype=np.float32)
        else:
            return np.array([norm_bird_y, norm_dist_x, norm_target_y], dtype=np.float32)

    def step(self, action):
        terminated = False
        
        if action == 1: self.bird_velocity = self.jump_velocity
        self.bird_velocity += self.gravity
        if self.bird_velocity > self.max_velocity: self.bird_velocity = self.max_velocity
        self.bird_y += self.bird_velocity
        
        self.frames_since_last_pipe += 1
        if self.frames_since_last_pipe > self.pipe_frequency:
            pipe_height = random.randint(-150, 150)
            bottom_y = int(self.screen_height / 2) + pipe_height + int(self.pipe_gap / 2)
            self.pipes.append({"x": self.screen_width, "bottom_y": bottom_y, "passed": False})
            self.frames_since_last_pipe = 0
            
        for pipe in self.pipes:
            pipe["x"] -= self.speed
            if pipe["x"] + 78 < self.bird_x and not pipe["passed"]:
                pipe["passed"] = True
                self.score += 1
                
        self.pipes = [p for p in self.pipes if p["x"] + 78 > 0]
        
        if self.bird_y >= self.ground_level or self.bird_y <= 0:
            terminated = True
            
        bird_rect = pygame.Rect(self.bird_x - 17, self.bird_y - 12, 34, 24)
        for pipe in self.pipes:
            top_pipe = pygame.Rect(pipe["x"], 0, 78, pipe["bottom_y"] - self.pipe_gap)
            bottom_pipe = pygame.Rect(pipe["x"], pipe["bottom_y"], 78, self.screen_height)
            if bird_rect.colliderect(top_pipe) or bird_rect.colliderect(bottom_pipe):
                terminated = True

        self.render()
        
        # Χειρισμός για να μπορείς να κλείσεις το παράθυρο με το (X)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        return self._get_obs(), 0, terminated, False, {"score": self.score}

    def render(self):
        self._load_assets()
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Flappy Bird AI - Evaluation Mode")
            self.clock = pygame.time.Clock()

        self.screen.blit(self.bg_img, (0, 0))
        for pipe in self.pipes:
            self.screen.blit(self.pipe_img, (pipe["x"], pipe["bottom_y"]))
            self.screen.blit(pygame.transform.flip(self.pipe_img, False, True), (pipe["x"], pipe["bottom_y"] - self.pipe_gap - self.pipe_img.get_height()))
            
        self.ground_move -= self.speed
        if abs(self.ground_move) > 35: self.ground_move = 0
        self.screen.blit(self.ground_img, (self.ground_move, self.ground_level))
        
        self.bird_counter += 1
        if self.bird_counter > 5:
            self.bird_counter = 0
            self.bird_index = (self.bird_index + 1) % 3
        bird_img = pygame.transform.rotate(self.bird_imgs[self.bird_index], self.bird_velocity * -2)
        self.screen.blit(bird_img, bird_img.get_rect(center=(int(self.bird_x), int(self.bird_y))))
        
        score_surface = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_surface, (int(self.screen_width / 2) - 80, 20))

        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


# --- ΛΟΓΙΚΗ ΜΕΝΟΥ ---
if __name__ == "__main__":
    BASE_DIR = "/home/nakos/Desktop/Εργασίες/Πτυχιακή/Flappy_Bird"
    
    # Λεξικό με τα ονόματα των φακέλων (προσαρμοσμένα στα ονόματα που μου έδωσες)
    EXPERIMENTS = {
        "1": {"name": "Exp_1_Basic", "obs": "basic"},
        "2": {"name": "Exp_2_Entropy_coef", "obs": "basic"},
        "3": {"name": "Exp_3_Survival_Reward", "obs": "basic"},
        "4": {"name": "Exp_4_Survival_Penalnty", "obs": "basic"},
        "5": {"name": "Exp_5_Velocity_Observation", "obs": "velocity"},
        "6": {"name": "Exp_6_Velocity_Observation_and_Survival_Reward", "obs": "velocity"},
        "7": {"name": "Exp_7_Velocity_Observation_and_Survival_Penalnty", "obs": "velocity"}
    }
    
    CHECKPOINTS = {
        "1": "50000",
        "2": "200000",
        "3": "500000",
        "4": "800000",
        "5": "1000000",
    }

    print("\n" + "="*50)
    print(" 🐦 FLAPPY BIRD AI - CONTROL ROOM 🐦 ")
    print("="*50)
    print("Επίλεξε Πείραμα για να τρέξεις:")
    print("1. Basic PPO (Χωρίς Entropy, 3 Inputs)")
    print("2. Με Entropy Coef (Περιέργεια, 3 Inputs)")
    print("3. Με Survival Reward (+0.1 ανά frame)")
    print("4. Με Survival Penalty (-0.1 ανά frame)")
    print("5. Με Velocity Observation (4 Inputs)")
    print("6. Με Velocity & Survival Reward")
    print("7. Με Velocity & Survival Penalty")
    print("="*50)
    
    exp_choice = input("Πληκτρολόγησε τον αριθμό του πειράματος (1-7): ")
    if exp_choice not in EXPERIMENTS:
        print("Λάθος επιλογή. Έξοδος...")
        sys.exit()
        
    print("\nΕπίλεξε Checkpoint (Βήμα Εκπαίδευσης):")
    print("1. 50.000 Steps")
    print("2. 200.000 Steps")
    print("3. 500.000 Steps")
    print("4. 800.000 Steps")
    print("5. 1.000.000 Steps (Τελικό)")
    print("6. Φόρτωση του 'best_model' (αν υπάρχει)")
    print("="*50)
    
    ckpt_choice = input("Πληκτρολόγησε τον αριθμό (1-6): ")
    
    folder_name = EXPERIMENTS[exp_choice]["name"]
    obs_mode = EXPERIMENTS[exp_choice]["obs"]
    
    if ckpt_choice == "6":
        model_file = "best_model"
    elif ckpt_choice in CHECKPOINTS:
        model_file = f"model_step_{CHECKPOINTS[ckpt_choice]}"
    else:
        print("Λάθος επιλογή. Έξοδος...")
        sys.exit()

    model_path = os.path.join(BASE_DIR, folder_name, "saved_models", model_file)
    
    if not os.path.exists(model_path + ".zip"):
        print(f"\n[ΣΦΑΛΜΑ] Το μοντέλο δεν βρέθηκε στη διαδρομή:\n{model_path}.zip")
        print("Μήπως δεν έχει ολοκληρωθεί ακόμα η εκπαίδευση αυτού του checkpoint;")
        sys.exit()

    print(f"\nΦόρτωση μοντέλου: {folder_name} -> {model_file}...")
    env = EvaluatorFlappyEnv(obs_mode=obs_mode)
    model = PPO.load(model_path)
    
    print("Το περιβάλλον ξεκινά! (Κλείσε το παράθυρο του παιχνιδιού για έξοδο)")
    
    # Ατέρμονο Loop για να παίζει συνεχόμενα
    while True:
        obs, _ = env.reset()
        terminated = False
        while not terminated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, _, info = env.step(action)