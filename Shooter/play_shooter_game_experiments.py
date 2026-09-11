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

# --- Constants & Level Map ---
TILE_SIZE = 40
COLS, ROWS = 30, 20
SCREEN_WIDTH = COLS * TILE_SIZE
SCREEN_HEIGHT = ROWS * TILE_SIZE

BG_COLOR = (30, 30, 30)
TILE_COLOR = (45, 45, 45)
WALL_COLOR = (240, 240, 240)
PLAYER_COLOR = (50, 150, 255)
ENEMY_COLOR = (255, 50, 50)
BULLET_PLAYER = (100, 255, 100)
BULLET_ENEMY = (255, 150, 50)

PLAYER_SPEED = 4.0 
ENEMY_SPEED = 2.5  
PLAYER_RADIUS_SHOOT = 250
ENEMY_RADIUS_SHOOT = 300
SHOOT_COOLDOWN_PLAYER = 400 
SHOOT_COOLDOWN_ENEMY = 700  
ENTITY_SIZE = 12 

LEVEL_MAP = [
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
for y in range(ROWS):
    for x in range(COLS):
        if LEVEL_MAP[y][x] == 1:
            walls.append(pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE))

# --- A* Logic ---
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
            if 0 <= neighbor[0] < COLS and 0 <= neighbor[1] < ROWS:
                if LEVEL_MAP[neighbor[1]][neighbor[0]] == 1: continue 
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
    return []

# --- Entities ---
class Entity:
    def __init__(self, x, y, speed, color):
        self.pos = Vector2(x, y)
        self.speed = speed
        self.radius = ENTITY_SIZE
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
        self.pos.x = max(self.radius, min(SCREEN_WIDTH - self.radius, self.pos.x))
        self.pos.y = max(self.radius, min(SCREEN_HEIGHT - self.radius, self.pos.y))

class Player(Entity):
    def __init__(self):
        super().__init__(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, PLAYER_SPEED, PLAYER_COLOR)
        self.radius = 10 

class Enemy(Entity):
    def __init__(self, x, y):
        super().__init__(x, y, ENEMY_SPEED, ENEMY_COLOR)
        self.path = []
        self.last_path_update = 0
        self.vx = 0.0
        self.vy = 0.0

    def update(self, player, enemies, bullets, current_time):
        if current_time - self.last_path_update > 500:
            start_grid = (int(self.pos.x // TILE_SIZE), int(self.pos.y // TILE_SIZE))
            target_grid = (int(player.pos.x // TILE_SIZE), int(player.pos.y // TILE_SIZE))
            self.path = astar(start_grid, target_grid)
            self.last_path_update = current_time + random.randint(0, 100)

        dx, dy = 0, 0
        if self.path:
            next_node = self.path[0]
            target_pos = Vector2(next_node[0] * TILE_SIZE + TILE_SIZE/2, next_node[1] * TILE_SIZE + TILE_SIZE/2)
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
                    
        if current_time - self.last_shot >= SHOOT_COOLDOWN_ENEMY:
            if self.pos.distance_to(player.pos) <= ENEMY_RADIUS_SHOOT:
                direction = (player.pos - self.pos).normalize()
                bullets.append(Bullet(self.pos.x, self.pos.y, direction, False, self))
                self.last_shot = current_time

class Bullet:
    def __init__(self, x, y, direction, is_player, owner=None):
        self.pos = Vector2(x, y)
        self.direction = direction
        self.speed = 8.0 
        self.is_player = is_player
        self.color = BULLET_PLAYER if is_player else BULLET_ENEMY
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
        if not (0 <= self.pos.x <= SCREEN_WIDTH and 0 <= self.pos.y <= SCREEN_HEIGHT):
            self.active = False


# --- ΕΝΙΑΙΟ ΠΕΡΙΒΑΛΛΟΝ ΓΙΑ ΑΞΙΟΛΟΓΗΣΗ ---
class EvaluatorShooterEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, vision_mode="static"):
        super().__init__()
        self.vision_mode = vision_mode
        self.action_space = spaces.Discrete(9)
        
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
        for x in range(COLS):
            if LEVEL_MAP[0][x] == 0: border_tiles.append((x, 0))
            if LEVEL_MAP[ROWS-1][x] == 0: border_tiles.append((x, ROWS-1))
        for y in range(ROWS):
            if LEVEL_MAP[y][0] == 0: border_tiles.append((0, y))
            if LEVEL_MAP[y][COLS-1] == 0: border_tiles.append((COLS-1, y))

        spawned = 0
        while spawned < count:
            spawn_tile = random.choice(border_tiles)
            spawn_pos = Vector2(spawn_tile[0]*TILE_SIZE + TILE_SIZE/2, spawn_tile[1]*TILE_SIZE + TILE_SIZE/2)
            if spawn_pos.distance_to(self.player.pos) > TILE_SIZE * 3:
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
            while dist < SCREEN_WIDTH and not hit:
                current += ray_dir * 5
                dist += 5
                if not (0 <= current.x <= SCREEN_WIDTH and 0 <= current.y <= SCREEN_HEIGHT):
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
                dx = (sorted_enemies[i].pos.x - self.player.pos.x) / SCREEN_WIDTH
                dy = (sorted_enemies[i].pos.y - self.player.pos.y) / SCREEN_HEIGHT
                obs.extend([dx, dy])
                if self.vision_mode == "kinematic":
                    obs.extend([sorted_enemies[i].vx / ENEMY_SPEED, sorted_enemies[i].vy / ENEMY_SPEED])
            else:
                obs.extend([0.0, 0.0])
                if self.vision_mode == "kinematic": obs.extend([0.0, 0.0])

        enemy_bullets = [b for b in self.bullets if not b.is_player]
        sorted_bullets = sorted(enemy_bullets, key=lambda b: self.player.pos.distance_to(b.pos))
        for i in range(3):
            if i < len(sorted_bullets):
                dx = (sorted_bullets[i].pos.x - self.player.pos.x) / SCREEN_WIDTH
                dy = (sorted_bullets[i].pos.y - self.player.pos.y) / SCREEN_HEIGHT
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

        if self.sim_time - self.player.last_shot >= SHOOT_COOLDOWN_PLAYER and self.enemies:
            closest_enemy = min(self.enemies, key=lambda e: self.player.pos.distance_to(e.pos))
            if self.player.pos.distance_to(closest_enemy.pos) <= PLAYER_RADIUS_SHOOT:
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
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Shooter AI - Evaluation Mode")
            self.clock = pygame.time.Clock()
            pygame.font.init()
            self.font = pygame.font.SysFont('Bauhaus 93', 40)

        self.screen.fill(BG_COLOR)
        
        for y in range(ROWS):
            for x in range(COLS):
                if LEVEL_MAP[y][x] == 0:
                    pygame.draw.rect(self.screen, TILE_COLOR, (x*TILE_SIZE+1, y*TILE_SIZE+1, TILE_SIZE-2, TILE_SIZE-2))

        wall_padding = 4
        for wall in walls:
            pygame.draw.rect(self.screen, WALL_COLOR, (wall.x + wall_padding, wall.y + wall_padding, TILE_SIZE - wall_padding*2, TILE_SIZE - wall_padding*2))
            
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


# --- ΛΟΓΙΚΗ ΜΕΝΟΥ ---
if __name__ == "__main__":
    BASE_DIR = "/home/nakos/Desktop/Εργασίες/Πτυχιακή/Shooter_Game"
    
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

    print("\n" + "="*50)
    print(" 🔫 SHOOTER AI - CONTROL ROOM 🔫 ")
    print("="*50)
    print("Επίλεξε Πείραμα για να τρέξεις:")
    print("1. Basic PPO (Στατικό Ραντάρ, 64x64)")
    print("2. Network Scaling (Στατικό Ραντάρ, 256x256)")
    print("3. The Camper (Θετική Ανταμοιβή Επιβίωσης)")
    print("4. The Rusher (Αρνητική Ποινή Χρόνου)")
    print("5. Kinematic Vision (Προσθήκη Ταχύτητας Σφαιρών)")
    print("6. Kinematics + Survival Reward")
    print("7. The Mastermind (Kinematics + Survival Penalty)")
    print("="*50)
    
    exp_choice = input("Πληκτρολόγησε τον αριθμό του πειράματος (1-7): ")
    if exp_choice not in EXPERIMENTS:
        print("Λάθος επιλογή. Έξοδος...")
        sys.exit()
        
    print("\nΕπίλεξε Checkpoint (Βήμα Εκπαίδευσης):")
    print("1. 400.000 Steps")
    print("2. 800.000 Steps")
    print("3. 1.200.000 Steps")
    print("4. 1.600.000 Steps")
    print("5. 2.000.000 Steps (Τελικό)")
    print("6. Φόρτωση του 'best_model' (αν υπάρχει)")
    print("="*50)
    
    ckpt_choice = input("Πληκτρολόγησε τον αριθμό (1-6): ")
    
    folder_name = EXPERIMENTS[exp_choice]["name"]
    vision_mode = EXPERIMENTS[exp_choice]["vision"]
    
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
    env = EvaluatorShooterEnv(vision_mode=vision_mode)
    model = PPO.load(model_path)
    
    print("Το περιβάλλον ξεκινά! (Κλείσε το παράθυρο του παιχνιδιού για έξοδο)")
    
    while True:
        obs, _ = env.reset()
        terminated = False
        while not terminated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, _, info = env.step(action)