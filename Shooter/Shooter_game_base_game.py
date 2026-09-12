import pygame
import sys
import random
import heapq
from pygame.math import Vector2

pygame.init()
fps = 60
clock = pygame.time.Clock()

Tile_size = 40
Columns, Rows = 30, 20
Screen_width = Columns * Tile_size
Screen_height = Rows * Tile_size

screen = pygame.display.set_mode((Screen_width, Screen_height))
pygame.display.set_caption("Auto-Shoot Survival")
font = pygame.font.SysFont('Bauhaus 93', 40)

# Colors
bg_clour = (30, 30, 30)
tile_colour = (45, 45, 45)
wall_colour = (240, 240, 240)
player_colour = (50, 150, 255)
enemy_colour = (255, 50, 50)
player_bullet = (100, 255, 100)
enemy_bullet = (255, 150, 50)

# Game Variables
Player_speed = 4.0 
Enemy_speed = 2.5  
Player_shooting_radius = 250
Enemy_shooting_radius = 300
Player_shot_cooldown = 400 # ms
Enemy_shot_cooldown = 700  # ms
Enemy_size = 12 

# Map Layout (1 = Wall, 0 = Floor) 
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

# This generates wall rectangles for collision
walls = []
for y in range(Rows):
    for x in range(Columns):
        if Map[y][x] == 1:
            walls.append(pygame.Rect(x * Tile_size, y * Tile_size, Tile_size, Tile_size))

# A* Pathfinding 
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
                if Map[neighbor[1]][neighbor[0]] == 1:
                    continue 
                
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
    return []

class Entity:
    def __init__(self, x, y, color, speed):
        self.pos = Vector2(x, y)
        self.color = color
        self.speed = speed
        self.radius = Enemy_size
        self.last_shot = 0
        
    def get_rect(self):
        return pygame.Rect(self.pos.x - self.radius, self.pos.y - self.radius, self.radius*2, self.radius*2)

    def move_with_collision(self, dx, dy, other_entities):
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

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.pos.x), int(self.pos.y)), self.radius)

class Player(Entity):
    def __init__(self):
        super().__init__(Screen_width // 2, Screen_height // 2, player_colour, Player_speed)
        self.radius = 10 # Size of the player
        
    def update(self, enemies, bullets):
        keys = pygame.key.get_pressed()
        dx, dy = 0, 0
        if keys[pygame.K_w] or keys[pygame.K_UP]: dy -= self.speed
        if keys[pygame.K_s] or keys[pygame.K_DOWN]: dy += self.speed
        if keys[pygame.K_a] or keys[pygame.K_LEFT]: dx -= self.speed
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: dx += self.speed
        
        if dx != 0 and dy != 0:
            dx *= 0.7071
            dy *= 0.7071
            
        self.move_with_collision(dx, dy, enemies)
        
        now = pygame.time.get_ticks()
        if now - self.last_shot >= Player_shot_cooldown and enemies:
            closest_enemy = min(enemies, key=lambda e: self.pos.distance_to(e.pos))
            if self.pos.distance_to(closest_enemy.pos) <= Player_shooting_radius:
                direction = (closest_enemy.pos - self.pos).normalize()
                bullets.append(Bullet(self.pos.x, self.pos.y, direction, is_player=True, owner=self))
                self.last_shot = now

class Enemy(Entity):
    def __init__(self, x, y):
        super().__init__(x, y, enemy_colour, Enemy_speed)
        self.path = []
        self.last_path_update = 0

    def update(self, player, enemies, bullets):
        now = pygame.time.get_ticks()
        
        if now - self.last_path_update > 500:
            start_grid = (int(self.pos.x // Tile_size), int(self.pos.y // Tile_size))
            target_grid = (int(player.pos.x // Tile_size), int(player.pos.y // Tile_size))
            self.path = astar(start_grid, target_grid)
            self.last_path_update = now + random.randint(0, 100)

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

        self.move_with_collision(dx, dy, enemies)
        
        for other in enemies:
            if other != self:
                dist = self.pos.distance_to(other.pos)
                min_dist = self.radius * 2
                if dist < min_dist and dist > 0:
                    overlap = min_dist - dist
                    push_dir = (self.pos - other.pos).normalize()
                    self.pos += push_dir * (overlap / 2)
                    
        if now - self.last_shot >= Enemy_shot_cooldown:
            if self.pos.distance_to(player.pos) <= Enemy_shooting_radius:
                direction = (player.pos - self.pos).normalize()
                bullets.append(Bullet(self.pos.x, self.pos.y, direction, is_player=False, owner=self))
                self.last_shot = now

class Bullet:
    def __init__(self, x, y, direction, is_player, owner=None):
        self.pos = Vector2(x, y)
        self.direction = direction
        self.speed = 8
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

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.pos.x), int(self.pos.y)), self.radius)

class Game:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.player = Player()
        self.enemies = []
        self.bullets = []
        self.level = 1           
        self.score = 0          
        self.base_enemy_count = 3
        self.state = "PLAYING"
        self.transition_timer = 0
        self.spawn_enemies()

    def spawn_enemies(self):
        count = self.base_enemy_count + (self.level // 2) * 2
        spawned = 0
        
        border_tiles = []
        for x in range(Columns):
            if Map[0][x] == 0: border_tiles.append((x, 0))
            if Map[Rows-1][x] == 0: border_tiles.append((x, Rows-1))
        for y in range(Rows):
            if Map[y][0] == 0: border_tiles.append((0, y))
            if Map[y][Columns-1] == 0: border_tiles.append((Columns-1, y))

        while spawned < count:
            spawn_tile = random.choice(border_tiles)
            spawn_pos = Vector2(spawn_tile[0]*Tile_size + Tile_size/2, spawn_tile[1]*Tile_size + Tile_size/2)
            
            if spawn_pos.distance_to(self.player.pos) > Tile_size * 3:
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

        self.player.update(self.enemies, self.bullets)

        for enemy in self.enemies:
            enemy.update(self.player, self.enemies, self.bullets)
            
        for bullet in self.bullets[:]:

            if not bullet.is_player and bullet.owner not in self.enemies:
                bullet.active = False
            
            bullet.update()
            
            if not bullet.active:
                if bullet in self.bullets:
                    self.bullets.remove(bullet)
                continue

            bullet_rect = pygame.Rect(bullet.pos.x - bullet.radius, bullet.pos.y - bullet.radius, bullet.radius*2, bullet.radius*2)
            
            if bullet.is_player:
                for enemy in self.enemies[:]:
                    enemy_rect = enemy.get_rect()
                    if bullet_rect.colliderect(enemy_rect):
                        self.enemies.remove(enemy)
                        self.score += 1
                        bullet.active = False
                        break
            else:
                player_rect = self.player.get_rect()
                if bullet_rect.colliderect(player_rect):
                    if self.state != "GAMEOVER":
                        self.state = "GAMEOVER"
                    
            if not bullet.active and bullet in self.bullets:
                self.bullets.remove(bullet)

        if len(self.enemies) == 0 and self.state == "PLAYING":
            self.state = "TRANSITION"
            self.transition_timer = pygame.time.get_ticks()

    def draw(self):
        screen.fill(bg_clour)
        
        for y in range(Rows):
            for x in range(Columns):
                if Map[y][x] == 0:
                    pygame.draw.rect(screen, tile_colour, (x*Tile_size+1, y*Tile_size+1, Tile_size-2, Tile_size-2))

        wall_padding = 4
        for wall in walls:
            pygame.draw.rect(screen, wall_colour, (wall.x + wall_padding, wall.y + wall_padding, Tile_size - wall_padding*2, Tile_size - wall_padding*2))
            
        for bullet in self.bullets:
            bullet.draw(screen)
        for enemy in self.enemies:
            enemy.draw(screen)
        self.player.draw(screen)
        
        score_text = font.render(f'Score: {self.score} | Wave: {self.level}', True, (210, 210, 210))
        screen.blit(score_text, (10, 10))
        
        if self.state == "TRANSITION":
            trans_text = font.render('WAVE CLEARED! Get ready...', True, (255, 255, 0))
            rect = trans_text.get_rect(center=(Screen_width/2, Screen_height/2))
            screen.blit(trans_text, rect)
            
        elif self.state == "GAMEOVER":
            over_text = font.render('GAME OVER - Press SPACE to Restart', True, (255, 50, 50))
            rect = over_text.get_rect(center=(Screen_width/2, Screen_height/2))
            screen.blit(over_text, rect)

game = Game()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if not running:
        break
            
    game.update()
    game.draw()
    pygame.display.flip()
    clock.tick(fps)

pygame.quit()
sys.exit()