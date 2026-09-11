import pygame
import sys
import random
import heapq
import os
import csv
from pygame.math import Vector2

# --- Setup & Constants ---
pygame.init()
fps = 60
clock = pygame.time.Clock()

TILE_SIZE = 40
COLS, ROWS = 30, 20
SCREEN_WIDTH = COLS * TILE_SIZE
SCREEN_HEIGHT = ROWS * TILE_SIZE

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Auto-Shoot Survival")
font = pygame.font.SysFont('Bauhaus 93', 40)

# Colors
BG_COLOR = (30, 30, 30)
TILE_COLOR = (45, 45, 45)
WALL_COLOR = (240, 240, 240)
PLAYER_COLOR = (50, 150, 255)
ENEMY_COLOR = (255, 50, 50)
BULLET_PLAYER = (100, 255, 100)
BULLET_ENEMY = (255, 150, 50)

# Game Balance Variables
PLAYER_SPEED = 4.0 # Λίγο πιο γρήγορο
ENEMY_SPEED = 2.5  # Λίγο πιο γρήγορο
PLAYER_RADIUS_SHOOT = 250
ENEMY_RADIUS_SHOOT = 300
SHOOT_COOLDOWN_PLAYER = 400 # ms
SHOOT_COOLDOWN_ENEMY = 700  # ms
ENTITY_SIZE = 12 # Default size για εχθρούς

# --- Map Layout (1 = Wall, 0 = Floor) --- 
# Μεγαλύτερος χάρτης 30x20
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

# Generate Wall Rects for Collision
walls = []
for y in range(ROWS):
    for x in range(COLS):
        if LEVEL_MAP[y][x] == 1:
            walls.append(pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE))

# --- A* Pathfinding Logic ---
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
            if 0 <= neighbor[0] < COLS and 0 <= neighbor[1] < ROWS:
                if LEVEL_MAP[neighbor[1]][neighbor[0]] == 1:
                    continue 
                
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
    return []

# --- Classes ---
class Entity:
    def __init__(self, x, y, color, speed):
        self.pos = Vector2(x, y)
        self.color = color
        self.speed = speed
        self.radius = ENTITY_SIZE
        self.last_shot = 0
        
    def get_rect(self):
        return pygame.Rect(self.pos.x - self.radius, self.pos.y - self.radius, self.radius*2, self.radius*2)

    def move_with_collision(self, dx, dy, other_entities):
        # Wall Collision X
        self.pos.x += dx
        rect = self.get_rect()
        for wall in walls:
            if rect.colliderect(wall):
                if dx > 0: self.pos.x = wall.left - self.radius
                if dx < 0: self.pos.x = wall.right + self.radius
        
        # Wall Collision Y
        self.pos.y += dy
        rect = self.get_rect()
        for wall in walls:
            if rect.colliderect(wall):
                if dy > 0: self.pos.y = wall.top - self.radius
                if dy < 0: self.pos.y = wall.bottom + self.radius
                
        # Screen bounds (soft clamp)
        self.pos.x = max(self.radius, min(SCREEN_WIDTH - self.radius, self.pos.x))
        self.pos.y = max(self.radius, min(SCREEN_HEIGHT - self.radius, self.pos.y))

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.pos.x), int(self.pos.y)), self.radius)

class Player(Entity):
    def __init__(self):
        super().__init__(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, PLAYER_COLOR, PLAYER_SPEED)
        self.radius = 10 # Μικρότερο hitbox (radius) για τον παίκτη
        
    def update(self, enemies, bullets):
        # Keyboard Movement
        keys = pygame.key.get_pressed()
        dx, dy = 0, 0
        if keys[pygame.K_w] or keys[pygame.K_UP]: dy -= self.speed
        if keys[pygame.K_s] or keys[pygame.K_DOWN]: dy += self.speed
        if keys[pygame.K_a] or keys[pygame.K_LEFT]: dx -= self.speed
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: dx += self.speed
        
        # Normalize diagonal speed
        if dx != 0 and dy != 0:
            dx *= 0.7071
            dy *= 0.7071
            
        self.move_with_collision(dx, dy, enemies)
        
        # Auto-shoot closest enemy
        now = pygame.time.get_ticks()
        if now - self.last_shot >= SHOOT_COOLDOWN_PLAYER and enemies:
            closest_enemy = min(enemies, key=lambda e: self.pos.distance_to(e.pos))
            if self.pos.distance_to(closest_enemy.pos) <= PLAYER_RADIUS_SHOOT:
                direction = (closest_enemy.pos - self.pos).normalize()
                bullets.append(Bullet(self.pos.x, self.pos.y, direction, is_player=True, owner=self))
                self.last_shot = now

class Enemy(Entity):
    def __init__(self, x, y):
        super().__init__(x, y, ENEMY_COLOR, ENEMY_SPEED)
        self.path = []
        self.last_path_update = 0

    def update(self, player, enemies, bullets):
        now = pygame.time.get_ticks()
        
        # Calculate Path every 500ms
        if now - self.last_path_update > 500:
            start_grid = (int(self.pos.x // TILE_SIZE), int(self.pos.y // TILE_SIZE))
            target_grid = (int(player.pos.x // TILE_SIZE), int(player.pos.y // TILE_SIZE))
            self.path = astar(start_grid, target_grid)
            self.last_path_update = now + random.randint(0, 100)

        # Movement Execution
        dx, dy = 0, 0
        if self.path:
            next_node = self.path[0]
            target_pos = Vector2(next_node[0] * TILE_SIZE + TILE_SIZE/2, next_node[1] * TILE_SIZE + TILE_SIZE/2)
            
            if self.pos.distance_to(target_pos) < self.speed:
                self.path.pop(0)
            else:
                direction = (target_pos - self.pos).normalize()
                dx, dy = direction.x * self.speed, direction.y * self.speed
        else:
            if self.pos.distance_to(player.pos) > self.radius:
                direction = (player.pos - self.pos).normalize()
                dx, dy = direction.x * self.speed, direction.y * self.speed

        self.move_with_collision(dx, dy, enemies)
        
        # Separation
        for other in enemies:
            if other != self:
                dist = self.pos.distance_to(other.pos)
                min_dist = self.radius * 2
                if dist < min_dist and dist > 0:
                    overlap = min_dist - dist
                    push_dir = (self.pos - other.pos).normalize()
                    self.pos += push_dir * (overlap / 2)
                    
        # Auto-shoot player
        if now - self.last_shot >= SHOOT_COOLDOWN_ENEMY:
            if self.pos.distance_to(player.pos) <= ENEMY_RADIUS_SHOOT:
                direction = (player.pos - self.pos).normalize()
                bullets.append(Bullet(self.pos.x, self.pos.y, direction, is_player=False, owner=self))
                self.last_shot = now

class Bullet:
    def __init__(self, x, y, direction, is_player, owner=None):
        self.pos = Vector2(x, y)
        self.direction = direction
        self.speed = 8
        self.is_player = is_player
        self.color = BULLET_PLAYER if is_player else BULLET_ENEMY
        self.radius = 4
        self.active = True
        self.owner = owner # Κρατάμε ποιος έριξε τη σφαίρα

    def update(self):
        self.pos += self.direction * self.speed
        
        rect = pygame.Rect(self.pos.x - self.radius, self.pos.y - self.radius, self.radius*2, self.radius*2)
        for wall in walls:
            if rect.colliderect(wall):
                self.active = False
                break
                
        if not (0 <= self.pos.x <= SCREEN_WIDTH and 0 <= self.pos.y <= SCREEN_HEIGHT):
            self.active = False

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.pos.x), int(self.pos.y)), self.radius)

# --- Game Manager ---
class Game:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.player = Player()
        self.enemies = []
        self.bullets = []
        self.level = 1           # Το level κρατήθηκε εσωτερικά για να δυσκολεύει το παιχνίδι
        self.score = 0           # Προστεθηκε το Score!
        self.base_enemy_count = 3
        self.state = "PLAYING"
        self.transition_timer = 0
        self.spawn_enemies()

    def save_score_to_csv(self):
        filename = "shooter_game_csv.csv"
        file_exists = os.path.isfile(filename)
        try_num = 1
        
        # Αν υπάρχει ήδη, διαβάζουμε πόσες προσπάθειες έχουν γίνει
        if file_exists:
            with open(filename, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = [row for row in reader if row]
                if len(rows) > 0:
                    try_num = len(rows) # Επειδή η 1η γραμμή είναι το Header, το len = επόμενη προσπάθεια
                    
        # Προσθήκη του καινούριου σκορ
        with open(filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists or try_num == 1:
                writer.writerow(["Try", "Score"]) # Δημιουργία Header αν είναι το 1ο game
            writer.writerow([f"Try {try_num}", self.score])

    def spawn_enemies(self):
        count = self.base_enemy_count + (self.level // 2) * 2
        spawned = 0
        
        border_tiles = []
        for x in range(COLS):
            if LEVEL_MAP[0][x] == 0: border_tiles.append((x, 0))
            if LEVEL_MAP[ROWS-1][x] == 0: border_tiles.append((x, ROWS-1))
        for y in range(ROWS):
            if LEVEL_MAP[y][0] == 0: border_tiles.append((0, y))
            if LEVEL_MAP[y][COLS-1] == 0: border_tiles.append((COLS-1, y))

        while spawned < count:
            spawn_tile = random.choice(border_tiles)
            spawn_pos = Vector2(spawn_tile[0]*TILE_SIZE + TILE_SIZE/2, spawn_tile[1]*TILE_SIZE + TILE_SIZE/2)
            
            if spawn_pos.distance_to(self.player.pos) > TILE_SIZE * 3:
                self.enemies.append(Enemy(spawn_pos.x, spawn_pos.y))
                spawned += 1

    def update(self):
        if self.state == "GAMEOVER":
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                self.reset()
            return

        if self.state == "TRANSITION":
            now = pygame.time.get_ticks()
            if now - self.transition_timer >= 1000:
                self.level += 1
                self.spawn_enemies()
                self.state = "PLAYING"
            return

        # Update Player
        self.player.update(self.enemies, self.bullets)
        
        # Update Enemies
        for enemy in self.enemies:
            enemy.update(self.player, self.enemies, self.bullets)
            
        # Update Bullets & Collisions
        for bullet in self.bullets[:]:
            
            # Αν ο εχθρός που έριξε τη σφαίρα έχει πεθάνει, καταστρέφεται και η σφαίρα του
            if not bullet.is_player and bullet.owner not in self.enemies:
                bullet.active = False
            
            bullet.update()
            
            if not bullet.active:
                if bullet in self.bullets:
                    self.bullets.remove(bullet)
                continue # Πάμε στην επόμενη σφαίρα
            
            # Entity Collision check
            bullet_rect = pygame.Rect(bullet.pos.x - bullet.radius, bullet.pos.y - bullet.radius, bullet.radius*2, bullet.radius*2)
            
            if bullet.is_player:
                for enemy in self.enemies[:]:
                    enemy_rect = enemy.get_rect()
                    if bullet_rect.colliderect(enemy_rect):
                        self.enemies.remove(enemy)
                        self.score += 1 # Εδώ ανεβαίνει το σκορ!
                        bullet.active = False
                        break
            else:
                player_rect = self.player.get_rect()
                if bullet_rect.colliderect(player_rect):
                    # Μόλις πεθάνει ο παίκτης, αποθηκεύουμε το σκορ μία φορά
                    if self.state != "GAMEOVER":
                        self.state = "GAMEOVER"
                        self.save_score_to_csv()
                    
            if not bullet.active and bullet in self.bullets:
                self.bullets.remove(bullet)
                
        # Check Level Clear
        if len(self.enemies) == 0 and self.state == "PLAYING":
            self.state = "TRANSITION"
            self.transition_timer = pygame.time.get_ticks()

    def draw(self):
        screen.fill(BG_COLOR)
        
        for y in range(ROWS):
            for x in range(COLS):
                if LEVEL_MAP[y][x] == 0:
                    pygame.draw.rect(screen, TILE_COLOR, (x*TILE_SIZE+1, y*TILE_SIZE+1, TILE_SIZE-2, TILE_SIZE-2))

        wall_padding = 4
        for wall in walls:
            pygame.draw.rect(screen, WALL_COLOR, (wall.x + wall_padding, wall.y + wall_padding, TILE_SIZE - wall_padding*2, TILE_SIZE - wall_padding*2))
            
        for bullet in self.bullets:
            bullet.draw(screen)
        for enemy in self.enemies:
            enemy.draw(screen)
        self.player.draw(screen)
        
        # Αντί για Level, τώρα εμφανίζει το Score
        score_text = font.render(f'Score: {self.score}', True, (210, 210, 210))
        screen.blit(score_text, (10, 10))
        
        if self.state == "TRANSITION":
            trans_text = font.render('WAVE CLEARED! Get ready...', True, (255, 255, 0))
            rect = trans_text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
            screen.blit(trans_text, rect)
            
        elif self.state == "GAMEOVER":
            over_text = font.render('GAME OVER - Press SPACE to Restart', True, (255, 50, 50))
            rect = over_text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
            screen.blit(over_text, rect)

# --- Main Loop ---
game = Game()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
            
    game.update()
    game.draw()
    
    pygame.display.flip()
    clock.tick(fps)