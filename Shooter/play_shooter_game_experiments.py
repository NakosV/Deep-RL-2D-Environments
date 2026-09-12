import pygame
import sys
import os
import random
import heapq
import numpy as np
from pygame.math import Vector2
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces

os.environ["OMP_NUM_THREADS"] = "1"

Tile_size = 40
Columns, Rows = 30, 20
Screen_width = Columns * Tile_size
Screen_height = Rows * Tile_size

bg_clour = (30, 30, 30)
tile_colour = (45, 45, 45)
wall_colour = (240, 240, 240)
player_colour = (50, 150, 255)
enemy_colour = (255, 50, 50)
player_bullet = (100, 255, 100)
enemy_bullet = (255, 150, 50)

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
def heuristic(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])

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
    def __init__(self, x, y, speed, color):
        self.pos = Vector2(x, y)
        self.speed = speed
        self.radius = Enemy_size
        self.color = color
        self.last_shot = 0
        
    def get_rect(self): return pygame.Rect(self.pos.x - self.radius, self.pos.y - self.radius, self.radius*2, self.radius*2)
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
        super().__init__(Screen_width // 2, Screen_height // 2, Player_speed, player_colour)
        self.radius = 10 

class Enemy(Entity):
    def __init__(self, x, y):
        super().__init__(x, y, Enemy_speed, enemy_colour)
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
            if self.pos.distance_to(target_pos) < self.speed: self.path.pop(0)
            else:
                direction = (target_pos - self.pos).normalize()
                dx, dy = direction.x * self.speed, direction.y * self.speed
        else:
            if self.pos.distance_to(player.pos) > self.radius:
                direction = (player.pos - self.pos).normalize()
                dx, dy = direction.x * self.speed, direction.y * self.speed

        self.vx, self.vy = dx, dy
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
                bullets.append(Bullet(self.pos.x, self.pos.y, direction, False, self))
                self.last_shot = current_time

class Bullet:
    def __init__(self, x, y, direction, is_player, owner=None):
        self.pos = Vector2(x, y)
        self.direction = direction
        self.speed = 8.0 
        self.is_player = is_player
        self.color = player_bullet if is_player else enemy_bullet
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

class EvaluatorShooterEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, vision_mode="static"):
        super().__init__()
        self.vision_mode = vision_mode
        self.action_space = spaces.Discrete(9)
        
        # It checks which type of vision the model that gets picked has and acts accordingly
        shape_size = 32 if vision_mode == "kinematic" else 20
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(shape_size,), dtype=np.float32)
        
        self.screen = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player = Player()
        self.enemies = []
        self.bullets = []
        self.level = 1
        self.base_enemy_count = 3
        self.spawn_enemies()
        self.sim_time = 0
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
                obs.extend([dx, dy])
                if self.vision_mode == "kinematic":
                    obs.extend([sorted_enemies[i].vx / Enemy_speed, sorted_enemies[i].vy / Enemy_speed])
            else:
                obs.extend([0.0, 0.0])
                if self.vision_mode == "kinematic": obs.extend([0.0, 0.0])

        enemy_bullets = [b for b in self.bullets if not b.is_player]
        sorted_bullets = sorted(enemy_bullets, key=lambda b: self.player.pos.distance_to(b.pos))
        for i in range(3):
            if i < len(sorted_bullets):
                dx = (sorted_bullets[i].pos.x - self.player.pos.x) / Screen_width
                dy = (sorted_bullets[i].pos.y - self.player.pos.y) / Screen_height
                obs.extend([dx, dy])
                if self.vision_mode == "kinematic":
                    vx = (sorted_bullets[i].direction.x * sorted_bullets[i].speed) / 8.0
                    vy = (sorted_bullets[i].direction.y * sorted_bullets[i].speed) / 8.0
                    obs.extend([vx, vy])
            else:
                obs.extend([0.0, 0.0])
                if self.vision_mode == "kinematic": obs.extend([0.0, 0.0])

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self.sim_time += 16 
        terminated = False

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
                self.bullets.append(Bullet(self.player.pos.x, self.player.pos.y, direction, True, self.player))
                self.player.last_shot = self.sim_time

        for enemy in self.enemies:
            enemy.update(self.player, self.enemies, self.bullets, self.sim_time)
            if self.player.pos.distance_to(enemy.pos) < self.player.radius + enemy.radius:
                terminated = True

        for bullet in self.bullets[:]:
            if not bullet.is_player and bullet.owner not in self.enemies: bullet.active = False
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
                        bullet.active = False
                        break
            else:
                if bullet_rect.colliderect(self.player.get_rect()):
                    terminated = True
                    
            if not bullet.active and bullet in self.bullets: self.bullets.remove(bullet)

        if len(self.enemies) == 0:
            self.level += 1
            self.spawn_enemies()

        self.render()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        return self._get_obs(), 0, terminated, False, {"score": self.score}

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((Screen_width, Screen_height))
            pygame.display.set_caption("Shooter AI - Evaluation Mode")
            self.clock = pygame.time.Clock()
            pygame.font.init()
            self.font = pygame.font.SysFont('Bauhaus 93', 40)

        self.screen.fill(bg_clour)
        
        for y in range(Rows):
            for x in range(Columns):
                if Map[y][x] == 0:
                    pygame.draw.rect(self.screen, tile_colour, (x*Tile_size+1, y*Tile_size+1, Tile_size-2, Tile_size-2))

        wall_padding = 4
        for wall in walls:
            pygame.draw.rect(self.screen, wall_colour, (wall.x + wall_padding, wall.y + wall_padding, Tile_size - wall_padding*2, Tile_size - wall_padding*2))
            
        for bullet in self.bullets:
            pygame.draw.circle(self.screen, bullet.color, (int(bullet.pos.x), int(bullet.pos.y)), bullet.radius)
        for enemy in self.enemies:
            pygame.draw.circle(self.screen, enemy.color, (int(enemy.pos.x), int(enemy.pos.y)), enemy.radius)
        
        pygame.draw.circle(self.screen, self.player.color, (int(self.player.pos.x), int(self.player.pos.y)), self.player.radius)
        
        score_text = self.font.render(f'Score: {self.score} | Wave: {self.level}', True, (210, 210, 210))
        self.screen.blit(score_text, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

if __name__ == "__main__":
    BASE_DIR = "/Place/Holder/Path_" # Replace it with where you have the models saved
    
    EXPERIMENTS = {
        "1": {"name": "Exp_1_Basic", "vision": "static"},
        "2": {"name": "Exp_2_Bigger_Network", "vision": "static"},
        "3": {"name": "Exp_3_Survival_Reward", "vision": "static"},
        "4": {"name": "Exp_4_Survival_Punishment", "vision": "static"},
        "5": {"name": "Exp_5_Kinematic_Vision", "vision": "kinematic"},
        "6": {"name": "Exp_6_Kinematic_Vision_&_Survival_Reward", "vision": "kinematic"},
        "7": {"name": "Exp_7_Kinematic_Vision_&_Survival_Punishment", "vision": "kinematic"}
    }
    
    CHECKPOINTS = {
        "1": "400000",
        "2": "800000",
        "3": "1200000",
        "4": "1600000",
        "5": "2000000",
    }

    print("1. Basic PPO")
    print("2. Network Scaling")
    print("3. Survival Reward")
    print("4. Survival Punishment")
    print("5. Kinematic Vision")
    print("6. Kinematics + Survival Reward")
    print("7. Kinematics + Survival Penalty")
    
    exp_choice = input("Choose the number of experiment that you want to try (1-7): ")
    if exp_choice not in EXPERIMENTS:
        sys.exit()
        
    print("1. 400.000 Steps")
    print("2. 800.000 Steps")
    print("3. 1.200.000 Steps")
    print("4. 1.600.000 Steps")
    print("5. 2.000.000 Steps")
    print("6. Best Model")
    
    ckpt_choice = input("Choose the number of model that you want to try (1-6): ")
    
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
        print(f"\n[Error] The model was not found at:\n{model_path}.zip")
        sys.exit()

    env = EvaluatorShooterEnv(vision_mode=vision_mode)
    model = PPO.load(model_path)
    
    while True:
        obs, _ = env.reset()
        terminated = False
        while not terminated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, _, info = env.step(action)