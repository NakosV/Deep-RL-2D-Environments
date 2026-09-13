import pygame
import os
import math
import numpy as np
from pygame.math import Vector2

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

os.environ["OMP_NUM_THREADS"] = "4"

class AdvancedSnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super(AdvancedSnakeEnv, self).__init__()
        self.cell_size = 40
        self.cell_number = 20
        self.width = self.cell_number * self.cell_size
        self.height = self.cell_number * self.cell_size
        
        # The AI can do 3 actions, and those actions are given a number starting from 0 and ending at 2
        # The actions are: continuing straight, turning right (clockwise), and turning left (counter-clockwise)
        self.action_space = spaces.Discrete(3)
        
        # The AI uses egocentric ray casting, utilizing 8 rays that cast relative to the snake's current moving direction (forward, right, backward, left, and diagonals). Through these rays, it can see the distance to the walls, the distance to the fruit, and the distance to its own body
        # Because the vision rotates with the snake's head, the AI no longer needs extra inputs to know its absolute moving direction, reducing the total inputs to 24
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(24,), dtype=np.float32)
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.snake_body = [Vector2(5, 10), Vector2(4, 10), Vector2(3, 10)]
        self.snake_direction = Vector2(1, 0)
        self.fruit_pos = self._randomize_fruit()
        self.score = 0
        self.frame_iteration = 0
        self.prev_distance = self._get_distance(self.snake_body[0], self.fruit_pos)
        return self._get_obs(), {}

    def _randomize_fruit(self):
        while True:
            pos = Vector2(np.random.randint(0, self.cell_number), np.random.randint(0, self.cell_number))
            if pos not in self.snake_body: return pos

    def _get_distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def _get_obs(self):
        head = self.snake_body[0]
        obs = []
        
        clock_wise = [Vector2(0, -1), Vector2(1, 0), Vector2(0, 1), Vector2(-1, 0)]
        idx = clock_wise.index(self.snake_direction)
        forward = clock_wise[idx]
        right = clock_wise[(idx + 1) % 4]
        backward = clock_wise[(idx + 2) % 4]
        left = clock_wise[(idx + 3) % 4]
        
        directions = [forward, forward + right, right, backward + right, backward, backward + left, left, forward + left]
        
        for ray_dir in directions:
            dist_wall, dist_apple, dist_body = 0, 0, 0
            step = 1
            current = head + ray_dir
            while 0 <= current.x < self.cell_number and 0 <= current.y < self.cell_number:
                if current == self.fruit_pos and dist_apple == 0: dist_apple = step
                if current in self.snake_body and dist_body == 0: dist_body = step
                current += ray_dir
                step += 1
            dist_wall = step
            obs.append(1.0 / dist_wall)
            obs.append(1.0 / dist_apple if dist_apple > 0 else 0.0)
            obs.append(1.0 / dist_body if dist_body > 0 else 0.0)
            
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self.frame_iteration += 1
        clock_wise = [Vector2(1, 0), Vector2(0, 1), Vector2(-1, 0), Vector2(0, -1)]
        idx = clock_wise.index(self.snake_direction)
        if action == 1: self.snake_direction = clock_wise[(idx + 1) % 4]
        elif action == 2: self.snake_direction = clock_wise[(idx - 1) % 4]

        body_copy = self.snake_body[:-1]
        body_copy.insert(0, body_copy[0] + self.snake_direction)
        self.snake_body = body_copy[:]
        
        head = self.snake_body[0]
        reward = 0
        terminated = False

        is_wall_collision = (head.x < 0 or head.x >= self.cell_number or head.y < 0 or head.y >= self.cell_number)
        is_body_collision = head in self.snake_body[1:]
        
        if is_wall_collision or is_body_collision or self.frame_iteration > 100 * len(self.snake_body):
            reward = -10.0 # Reward for dying
            terminated = True
            return self._get_obs(), reward, terminated, False, {"score": self.score}

        if head == self.fruit_pos:
            self.score += 1
            reward = 10.0 # Reward for eating a fruit
            self.snake_body.append(self.snake_body[-1])
            self.fruit_pos = self._randomize_fruit()
            self.frame_iteration = 0
        else:
            # The tactic from the third experiment makes a return. 
            # The program only gives the AI a negative reward every frame and no rewards to guide it towards food
            reward = -0.02 
            self.prev_distance = self._get_distance(head, self.fruit_pos)

        if self.render_mode == "human": self.render()
        return self._get_obs(), reward, terminated, False, {"score": self.score}

    def render(self):
        pass

    def close(self):
        if self.screen is not None: pygame.quit(); self.screen = None

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
                print(f"[Milestone] Αποθήκευση στο βήμα {m}")
        return True

if __name__ == "__main__":
    EXP_NAME = "Exp_7_Egocentric_No_Distance_Reward"
    base_dir = f"/PlaceHolderPath/{EXP_NAME}" # Replace the path to where you want the model to be saved
    log_dir = f"{base_dir}/tensorboard/"
    save_dir = f"{base_dir}/saved_models/"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    def make_train_env(): return Monitor(AdvancedSnakeEnv(render_mode=None))
    train_env = DummyVecEnv([make_train_env])
    eval_env = Monitor(AdvancedSnakeEnv(render_mode=None))
    
    eval_callback = EvalCallback(eval_env, best_model_save_path=save_dir, log_path=save_dir, eval_freq=10000, n_eval_episodes=5, deterministic=True)
    milestone_callback = MilestoneSaveCallback(save_dir=save_dir, milestones=[400000, 800000, 1200000, 1600000, 2000000])
    
    # The bigger Neural Network stays
    # Because of the sparse rewards the AI is given a bigger ent_coef to explore more
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    model = PPO("MlpPolicy", train_env, verbose=1, policy_kwargs=policy_kwargs, ent_coef=0.1)
    model.set_logger(new_logger)
    
    model.learn(total_timesteps=2000000, callback=CallbackList([eval_callback, milestone_callback]))
    model.save(f"{save_dir}/final_ppo_snake_ai")
