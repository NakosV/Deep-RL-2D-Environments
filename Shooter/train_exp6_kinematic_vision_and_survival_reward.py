import pygame
import random
import heapq
import os
import numpy as np
from pygame.math import Vector2
import logging

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

os.environ["OMP_NUM_THREADS"] = "4"

Tile_size = 40
Columns, Rows = 30, 20
Screen_width = Columns * Tile_size
Screen_height = Rows * Tile_size

Player_speed = 4.0 
Enemy_speed = 2.5  
Player_shooting_radius = 250
Enemy_shooting_radius = 300
Player_shot_cooldown = 400 
Enemy_shot_cooldown = 700  
Enemy_size = 12 

Map = [
    [0]*30,
    [0,0,0,0,0,0,1,1,0,0,0,1,0,0,1,1,0,0,0,0,0,0,1,1,0,0,1,1,0,0],
    [0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0],
    [0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,0,0],
    [0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0],
    [0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0],
    [0,1,1,1,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,1,0],
    [0]*30
]

walls = []
for y in range(Rows):
    for x in range(Columns):
        if Map[y][x] == 1:
            walls.append(pygame.Rect(x * Tile_size, y * Tile_size, Tile_size, Tile_size))

# A* Logic
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(start, goal):
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path
            
        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < Columns and 0 <= neighbor[1] < Rows:
                if Map[neighbor[1]][neighbor[0]] == 1: continue 
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
    return []

class Entity:
    def __init__(self, x, y, speed):
        self.pos = Vector2(x, y)
        self.speed = speed
        self.radius = Enemy_size
        self.last_shot = 0
        
    def get_rect(self):
        return pygame.Rect(self.pos.x - self.radius, self.pos.y - self.radius, self.radius*2, self.radius*2)

    def move_with_collision(self, dx, dy):
        self.pos.x += dx
        rect = self.get_rect()
        for wall in walls:
            if rect.colliderect(wall):
                if dx > 0: self.pos.x = wall.left - self.radius
                if dx < 0: self.pos.x = wall.right + self.radius
        
        self.pos.y += dy
        rect = self.get_rect()
        for wall in walls:
            if rect.colliderect(wall):
                if dy > 0: self.pos.y = wall.top - self.radius
                if dy < 0: self.pos.y = wall.bottom + self.radius
                
        self.pos.x = max(self.radius, min(Screen_width - self.radius, self.pos.x))
        self.pos.y = max(self.radius, min(Screen_height - self.radius, self.pos.y))

class Player(Entity):
    def __init__(self):
        super().__init__(Screen_width // 2, Screen_height // 2, Player_speed)
        self.radius = 10 

class Enemy(Entity):
    def __init__(self, x, y):
        super().__init__(x, y, Enemy_speed)
        self.path = []
        self.last_path_update = 0
        self.vx = 0.0
        self.vy = 0.0

    def update(self, player, enemies, bullets, current_time):
        if current_time - self.last_path_update > 500:
            start_grid = (int(self.pos.x // Tile_size), int(self.pos.y // Tile_size))
            target_grid = (int(player.pos.x // Tile_size), int(player.pos.y // Tile_size))
            self.path = astar(start_grid, target_grid)
            self.last_path_update = current_time + random.randint(0, 100)

        dx, dy = 0, 0
        if self.path:
            next_node = self.path[0]
            target_pos = Vector2(next_node[0] * Tile_size + Tile_size/2, next_node[1] * Tile_size + Tile_size/2)
            if self.pos.distance_to(target_pos) < self.speed:
                self.path.pop(0)
            else:
                direction = (target_pos - self.pos).normalize()
                dx, dy = direction.x * self.speed, direction.y * self.speed
        else:
            if self.pos.distance_to(player.pos) > self.radius:
                direction = (player.pos - self.pos).normalize()
                dx, dy = direction.x * self.speed, direction.y * self.speed

        self.vx = dx
        self.vy = dy
        self.move_with_collision(dx, dy)
        
        for other in enemies:
            if other != self:
                dist = self.pos.distance_to(other.pos)
                min_dist = self.radius * 2
                if dist < min_dist and dist > 0:
                    overlap = min_dist - dist
                    push_dir = (self.pos - other.pos).normalize()
                    self.pos += push_dir * (overlap / 2)
                    
        if current_time - self.last_shot >= Enemy_shot_cooldown:
            if self.pos.distance_to(player.pos) <= Enemy_shooting_radius:
                direction = (player.pos - self.pos).normalize()
                bullets.append(Bullet(self.pos.x, self.pos.y, direction, is_player=False, owner=self))
                self.last_shot = current_time

class Bullet:
    def __init__(self, x, y, direction, is_player, owner=None):
        self.pos = Vector2(x, y)
        self.direction = direction
        self.speed = 8.0
        self.is_player = is_player
        self.radius = 4
        self.active = True
        self.owner = owner 

    def update(self):
        self.pos += self.direction * self.speed
        rect = pygame.Rect(self.pos.x - self.radius, self.pos.y - self.radius, self.radius*2, self.radius*2)
        for wall in walls:
            if rect.colliderect(wall):
                self.active = False
                break
        if not (0 <= self.pos.x <= Screen_width and 0 <= self.pos.y <= Screen_height):
            self.active = False

class ShooterEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        
        # The AI can do 9 actions, and those actions are given a number starting from 0 and ending at 8
        # The actions are: staying still, going up, going up and right, going right, going down and right, going down, going down and left, going left and going up and left
        self.action_space = spaces.Discrete(9)
        
        # The AI uses ray casting, utilizing 8 rays looking at 8 different directions. It can also see the coordinates of the 3 closest enemies and the coordinates of the 3 closest bullets
        # From the fifth experiment and onward the AI can also see the direction in which the 3 closest enemies and bullets are moving
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(32,), dtype=np.float32)
        
        self.sim_time = 0 
        self.max_steps = 1500 
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player = Player()
        self.enemies = []
        self.bullets = []
        self.level = 1
        self.base_enemy_count = 3
        self.spawn_enemies()
        self.sim_time = 0
        self.steps = 0
        self.score = 0
        return self._get_obs(), {}

    def spawn_enemies(self):
        count = self.base_enemy_count + (self.level // 2) * 2
        border_tiles = []
        for x in range(Columns):
            if Map[0][x] == 0: border_tiles.append((x, 0))
            if Map[Rows-1][x] == 0: border_tiles.append((x, Rows-1))
        for y in range(Rows):
            if Map[y][0] == 0: border_tiles.append((0, y))
            if Map[y][Columns-1] == 0: border_tiles.append((Columns-1, y))

        spawned = 0
        while spawned < count:
            spawn_tile = random.choice(border_tiles)
            spawn_pos = Vector2(spawn_tile[0]*Tile_size + Tile_size/2, spawn_tile[1]*Tile_size + Tile_size/2)
            if spawn_pos.distance_to(self.player.pos) > Tile_size * 3:
                self.enemies.append(Enemy(spawn_pos.x, spawn_pos.y))
                spawned += 1

    def _get_obs(self):
        obs = []
        
        directions = [
            Vector2(0, -1), Vector2(1, -1).normalize(), Vector2(1, 0), Vector2(1, 1).normalize(), 
            Vector2(0, 1), Vector2(-1, 1).normalize(), Vector2(-1, 0), Vector2(-1, -1).normalize()
        ]
        
        for ray_dir in directions:
            current = Vector2(self.player.pos.x, self.player.pos.y)
            dist = 0
            hit = False
            while dist < Screen_width and not hit:
                current += ray_dir * 5
                dist += 5
                if not (0 <= current.x <= Screen_width and 0 <= current.y <= Screen_height):
                    hit = True
                    break
                for wall in walls:
                    if wall.collidepoint(current.x, current.y):
                        hit = True
                        break
            obs.append(1.0 / (dist/50.0) if dist > 0 else 0)

        sorted_enemies = sorted(self.enemies, key=lambda e: self.player.pos.distance_to(e.pos))
        for i in range(3):
            if i < len(sorted_enemies):
                dx = (sorted_enemies[i].pos.x - self.player.pos.x) / Screen_width
                dy = (sorted_enemies[i].pos.y - self.player.pos.y) / Screen_height
                vx = sorted_enemies[i].vx / Enemy_speed
                vy = sorted_enemies[i].vy / Enemy_speed
                obs.extend([dx, dy, vx, vy])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0])

        enemy_bullets = [b for b in self.bullets if not b.is_player]
        sorted_bullets = sorted(enemy_bullets, key=lambda b: self.player.pos.distance_to(b.pos))
        for i in range(3):
            if i < len(sorted_bullets):
                dx = (sorted_bullets[i].pos.x - self.player.pos.x) / Screen_width
                dy = (sorted_bullets[i].pos.y - self.player.pos.y) / Screen_height
                vx = (sorted_bullets[i].direction.x * sorted_bullets[i].speed) / 8.0 
                vy = (sorted_bullets[i].direction.y * sorted_bullets[i].speed) / 8.0
                obs.extend([dx, dy, vx, vy])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0])

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self.sim_time += 16 
        self.steps += 1
        reward = 0.01 # In the sixth experiment the program gives a small and positive reward to the AI
        terminated = False
        truncated = False

        dx, dy = 0, 0
        spd = self.player.speed
        diag = spd * 0.7071
        
        if action == 1: dy = -spd
        elif action == 2: dx, dy = diag, -diag
        elif action == 3: dx = spd
        elif action == 4: dx, dy = diag, diag
        elif action == 5: dy = spd
        elif action == 6: dx, dy = -diag, diag
        elif action == 7: dx = -spd
        elif action == 8: dx, dy = -diag, -diag

        self.player.move_with_collision(dx, dy)

        if self.sim_time - self.player.last_shot >= Player_shot_cooldown and self.enemies:
            closest_enemy = min(self.enemies, key=lambda e: self.player.pos.distance_to(e.pos))
            if self.player.pos.distance_to(closest_enemy.pos) <= Player_shooting_radius:
                direction = (closest_enemy.pos - self.player.pos).normalize()
                self.bullets.append(Bullet(self.player.pos.x, self.player.pos.y, direction, is_player=True, owner=self.player))
                self.player.last_shot = self.sim_time

        for enemy in self.enemies:
            enemy.update(self.player, self.enemies, self.bullets, self.sim_time)
            if self.player.pos.distance_to(enemy.pos) < self.player.radius + enemy.radius:
                reward = -10.0
                terminated = True
                
        for bullet in self.bullets[:]:
            if not bullet.is_player and bullet.owner not in self.enemies:
                bullet.active = False
            
            bullet.update()
            if not bullet.active:
                if bullet in self.bullets: self.bullets.remove(bullet)
                continue
                
            bullet_rect = pygame.Rect(bullet.pos.x - bullet.radius, bullet.pos.y - bullet.radius, bullet.radius*2, bullet.radius*2)
            
            if bullet.is_player:
                for enemy in self.enemies[:]:
                    if bullet_rect.colliderect(enemy.get_rect()):
                        self.enemies.remove(enemy)
                        self.score += 1
                        reward = 10.0 # Reward for kill
                        bullet.active = False
                        break
            else:
                if bullet_rect.colliderect(self.player.get_rect()):
                    reward = -10.0 # Reward for death
                    terminated = True
                    
            if not bullet.active and bullet in self.bullets:
                self.bullets.remove(bullet)

        # Wave Clear
        if len(self.enemies) == 0:
            self.level += 1
            self.spawn_enemies()

        if self.steps >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {"score": self.score}

    def render(self):
        pass

    def close(self):
        pass


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

if __name__ == "__main__":
    EXP_NAME = "Exp_6_Kinematic_Vision_&_Survival_Reward"
    base_dir = f"/PlaceHolderPath/{EXP_NAME}" # Replace it with where you want the model to be saved
    
    log_dir = f"{base_dir}/tensorboard/"
    save_dir = f"{base_dir}/saved_models/"
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_file = f"{base_dir}/training_logs.txt"
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')

    def make_train_env():
        return Monitor(ShooterEnv(render_mode=None))
    
    eval_env = Monitor(ShooterEnv(render_mode=None))
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
    
    milestones = [400000, 800000, 1200000, 1600000, 2000000]
    milestone_callback = MilestoneSaveCallback(save_dir=save_dir, milestones=milestones)
    
    callback_list = CallbackList([eval_callback, milestone_callback])

    # The change from the second experiment has been kept
    policy_kwargs = dict(net_arch=[256, 256])
    model = PPO("MlpPolicy", train_env, verbose=1, device="auto", policy_kwargs=policy_kwargs)
    model.set_logger(new_logger)

    model.learn(total_timesteps=2000000, callback=callback_list)
    
    model.save(f"{save_dir}/final_ppo_shooter_ai")
    train_env.close()