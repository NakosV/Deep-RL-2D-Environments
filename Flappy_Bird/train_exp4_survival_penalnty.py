import pygame
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

class FlappyBirdEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super(FlappyBirdEnv, self).__init__()
        
        self.screen_width = 864
        self.screen_height = 936
        self.speed = 4
        self.pipe_gap = 150
        self.pipe_frequency = 120 
        self.gravity = 0.5
        self.jump_velocity = -10
        self.max_velocity = 8
        self.ground_level = 768
        
        # The actions that the AI can do are 0 for Nothing and 1 for Jump
        self.action_space = spaces.Discrete(2)
        
        # The AI can see the height of the bird, the distance of the next pipe and the space of the hole at the next pipes
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        
        self.render_mode = render_mode

        # This function just resets things like the bird and the pipes 
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.bird_x = 100.0
        self.bird_y = self.screen_height / 2
        self.bird_velocity = 0.0
        
        self.pipes = [] 
        self.frames_since_last_pipe = self.pipe_frequency
        self.score = 0
        self.frame_iteration = 0
        
        return self._get_obs(), {}

    def _get_obs(self):
        next_pipe = None
        for pipe in self.pipes:
            if pipe["x"] + 78 > self.bird_x: # 78 is the width of a pipe
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
        
        return np.array([norm_bird_y, norm_dist_x, norm_target_y], dtype=np.float32)

    def step(self, action):
        self.frame_iteration += 1
        reward = -0.1  # The third experiment feeds the AI a small but negative reward every frame it is alive
        terminated = False
        
        if action == 1:
            self.bird_velocity = self.jump_velocity
            
        self.bird_velocity += self.gravity
        if self.bird_velocity > self.max_velocity:
            self.bird_velocity = self.max_velocity
            
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
                reward = 10.0 # The reward for passing through a pipe
                
        self.pipes = [p for p in self.pipes if p["x"] + 78 > 0]
        
        if self.bird_y >= self.ground_level or self.bird_y <= 0:
            terminated = True
            reward = -10.0
            
        bird_rect = pygame.Rect(self.bird_x - 17, self.bird_y - 12, 34, 24)
        
        for pipe in self.pipes:
            top_pipe_rect = pygame.Rect(pipe["x"], 0, 78, pipe["bottom_y"] - self.pipe_gap)
            bottom_pipe_rect = pygame.Rect(pipe["x"], pipe["bottom_y"], 78, self.screen_height)
            
            if bird_rect.colliderect(top_pipe_rect) or bird_rect.colliderect(bottom_pipe_rect):
                terminated = True
                reward = -10.0

        return self._get_obs(), reward, terminated, False, {"score": self.score}

    def render(self):
        pass

    def close(self):
        pass 

class Milestones(BaseCallback):
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
                print(f"[Milestone] The model was saved at step {m}")
        return True

if __name__ == "__main__":
    EXP_NAME = "Exp_4_Survival_Penalnty"
    base_dir = f"/PlaceHolder{EXP_NAME}" # Replace with the path that you want the model to be saved at
    
    log_dir = f"{base_dir}/tensorboard/"
    save_dir = f"{base_dir}/saved_models/"
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

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
    milestone_callback = Milestones(save_dir=save_dir, milestones=milestones)
    
    callback_list = CallbackList([eval_callback, milestone_callback])

    # Here there the changes from the second experiment are kept
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
    model = PPO("MlpPolicy", train_env, verbose=1, device="auto", policy_kwargs=policy_kwargs, ent_coef=0.01)
    model.set_logger(new_logger)
    
    model.learn(total_timesteps=1000000, callback=callback_list)
    
    model.save(f"{save_dir}/final_ppo_flappy_ai")
    train_env.close()