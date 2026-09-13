import pygame
import sys
import os
import numpy as np
from pygame.math import Vector2
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces

os.environ["OMP_NUM_THREADS"] = "1"


class EvaluatorSnakeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, vision_mode="absolute"):
        super().__init__()
        self.vision_mode = vision_mode
        self.cell_size = 40
        self.cell_number = 20
        self.width = self.cell_number * self.cell_size
        self.height = self.cell_number * self.cell_size
        self.action_space = spaces.Discrete(3)
        
        # The program sees what vision the experiment you want to run has, and adjusts things accordingly
        shape_size = 24 if vision_mode == "egocentric" else 28
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(shape_size,), dtype=np.float32)
        
        self.render_mode = "human"
        self.screen = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.snake_body = [Vector2(5, 10), Vector2(4, 10), Vector2(3, 10)]
        self.snake_direction = Vector2(1, 0)
        self.fruit_pos = self._randomize_fruit()
        self.score = 0
        return self._get_obs(), {}

    def _randomize_fruit(self):
        while True:
            pos = Vector2(np.random.randint(0, self.cell_number), np.random.randint(0, self.cell_number))
            if pos not in self.snake_body: return pos

    def _get_obs(self):
        head = self.snake_body[0]
        obs = []
        
        if self.vision_mode == "absolute":
            directions = [
                Vector2(0, -1), Vector2(1, -1), Vector2(1, 0), Vector2(1, 1), 
                Vector2(0, 1), Vector2(-1, 1), Vector2(-1, 0), Vector2(-1, -1)
            ]
        else:
            clock_wise = [Vector2(0, -1), Vector2(1, 0), Vector2(0, 1), Vector2(-1, 0)]
            idx = clock_wise.index(self.snake_direction)
            f = clock_wise[idx]
            r = clock_wise[(idx + 1) % 4]
            b = clock_wise[(idx + 2) % 4]
            l = clock_wise[(idx + 3) % 4]
            directions = [f, f+r, r, b+r, b, b+l, l, f+l]
            
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
            obs.extend([1.0 / dist_wall, 1.0 / dist_apple if dist_apple > 0 else 0.0, 1.0 / dist_body if dist_body > 0 else 0.0])
            
        if self.vision_mode == "absolute":
            obs.extend([
                1.0 if self.snake_direction == Vector2(0, -1) else 0.0,
                1.0 if self.snake_direction == Vector2(0, 1) else 0.0,
                1.0 if self.snake_direction == Vector2(-1, 0) else 0.0,
                1.0 if self.snake_direction == Vector2(1, 0) else 0.0
            ])
            
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        clock_wise = [Vector2(1, 0), Vector2(0, 1), Vector2(-1, 0), Vector2(0, -1)]
        idx = clock_wise.index(self.snake_direction)
        if action == 1: self.snake_direction = clock_wise[(idx + 1) % 4]
        elif action == 2: self.snake_direction = clock_wise[(idx - 1) % 4]

        body_copy = self.snake_body[:-1]
        body_copy.insert(0, body_copy[0] + self.snake_direction)
        self.snake_body = body_copy[:]
        
        head = self.snake_body[0]
        terminated = False

        is_wall_collision = (head.x < 0 or head.x >= self.cell_number or head.y < 0 or head.y >= self.cell_number)
        is_body_collision = head in self.snake_body[1:]
        
        if is_wall_collision or is_body_collision:
            terminated = True
            return self._get_obs(), 0, terminated, False, {"score": self.score}

        if head == self.fruit_pos:
            self.score += 1
            self.snake_body.append(self.snake_body[-1])
            self.fruit_pos = self._randomize_fruit()

        self.render()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        return self._get_obs(), 0, terminated, False, {"score": self.score}

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Snake AI - Evaluation Mode")
            self.clock = pygame.time.Clock()
            pygame.font.init()
            self.game_font = pygame.font.SysFont('Bauhaus 93', 60)

        self.screen.fill((20, 20, 20))
        shrink_amount = 5
        fruit_size = 24
        
        offset = int((self.cell_size - fruit_size) / 2)
        x_position = int(self.fruit_pos.x * self.cell_size) + offset
        y_position = int(self.fruit_pos.y * self.cell_size) + offset
        pygame.draw.rect(self.screen, (225, 50, 30), pygame.Rect(x_position, y_position, fruit_size, fruit_size))
        
        for index, block in enumerate(self.snake_body):
            x_pos = int(block.x * self.cell_size) + shrink_amount
            y_pos = int(block.y * self.cell_size) + shrink_amount
            b_width = self.cell_size - (shrink_amount * 2) 
            b_height = self.cell_size - (shrink_amount * 2)
            pygame.draw.rect(self.screen, (175, 180, 120), pygame.Rect(x_pos, y_pos, b_width, b_height))
            
            if index < len(self.snake_body) - 1:
                relation = self.snake_body[index] - self.snake_body[index + 1]
                rect = None
                if relation == Vector2(1, 0): rect = pygame.Rect(x_pos - (shrink_amount * 2), y_pos, shrink_amount * 2, b_height)
                elif relation == Vector2(-1, 0): rect = pygame.Rect(x_pos + b_width, y_pos, shrink_amount * 2, b_height)
                elif relation == Vector2(0, 1): rect = pygame.Rect(x_pos, y_pos - (shrink_amount * 2), b_width, shrink_amount * 2)
                elif relation == Vector2(0, -1): rect = pygame.Rect(x_pos, y_pos + b_height, b_width, shrink_amount * 2)
                if rect: pygame.draw.rect(self.screen, (175, 180, 120), rect)

        score_surface = self.game_font.render(str(self.score), True, pygame.Color('white'))
        self.screen.blit(score_surface, score_surface.get_rect(center=(self.width - 60, self.height - 40)))

        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

if __name__ == "__main__":
    BASE_DIR = "/Place_Holder_Path" # Repalace with the path that the models are saved at
    
    EXPERIMENTS = {
        "1": {"name": "Exp_1_Distance_Reward", "vision": "absolute"},
        "2": {"name": "Exp_2_Distance_Reward_And_High_Penalty", "vision": "absolute"},
        "3": {"name": "Exp_3_No_Distance_Reward_And_Higher_Penalty", "vision": "absolute"},
        "4": {"name": "Exp_4_Bigger_Gamma_and_Balanced_Rewards", "vision": "absolute"},
        "5": {"name": "Exp_5_Distance_Reward_and_Lower_Penalnty", "vision": "absolute"},
        "6": {"name": "Exp_6_Egocentric", "vision": "egocentric"},
        "7": {"name": "Exp_7_Egocentric_No_Distance_Reward", "vision": "egocentric"}
    }
    
    CHECKPOINTS = {
        "1": "400000",
        "2": "800000",
        "3": "1200000",
        "4": "1600000",
        "5": "2000000",
    }

    print("1. Distance Reward")
    print("2. Distance Reward with High Penalnty")
    print("3. No Distance Reward and Higher Penalnty")
    print("4. Bigger Gamma and more Balanced Rewards")
    print("5. Distance Reward with Lower Penalnty")
    print("6. Egocentric")
    print("7. Egocentric with No Distance Reward")
    print("="*50)
    
    exp_choice = input("Choose the number of the experiment (1-7): ")
    if exp_choice not in EXPERIMENTS:
        sys.exit()
        
    print("1. 400.000 Steps")
    print("2. 800.000 Steps")
    print("3. 1.200.000 Steps")
    print("4. 1.600.000 Steps")
    print("5. 2.000.000 Steps (Τελικό)")
    print("6. Best Model")
    print("="*50)
    
    ckpt_choice = input("Choose the number of the checkpoint (1-6): ")
    
    folder_name = EXPERIMENTS[exp_choice]["name"]
    vision_mode = EXPERIMENTS[exp_choice]["vision"]
    
    if ckpt_choice == "6":
        model_file = "best_model"
    elif ckpt_choice in CHECKPOINTS:
        model_file = f"model_step_{CHECKPOINTS[ckpt_choice]}"
    else:
        sys.exit()

    model_path = os.path.join(BASE_DIR, folder_name, "saved_models", model_file)
    
    if not os.path.exists(model_path + ".zip"):
        print(f"\n[ERROR] The model was not found at:\n{model_path}.zip")
        sys.exit()

    env = EvaluatorSnakeEnv(vision_mode=vision_mode)
    model = PPO.load(model_path)
    
    while True:
        obs, _ = env.reset()
        terminated = False
        while not terminated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, _, info = env.step(action)