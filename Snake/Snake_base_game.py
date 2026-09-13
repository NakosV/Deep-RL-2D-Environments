import pygame
import sys
from pygame.math import Vector2
import random

pygame.init()
cell_size = 40
cell_number = 20
clock = pygame.time.Clock()
game_font = pygame.font.SysFont('Bauhaus 93', 60)
fps = 60
timer = 125 # miliseconds

# visual
shrink_amount = 5
fruit_size = 24

# colors
snake_r = 175
snake_g = 180
snake_b = 120

fruit_r = 225
fruit_g = 50
fruit_b = 30

screen_r = 20
screen_g = 20
screen_b = 20

screen = pygame.display.set_mode((cell_number*cell_size, cell_number*cell_size))

class MAIN():
    def __init__(self):
        self.snake = SNAKE()
        self.fruit = FRUIT()
        self.game_won = False
        while self.fruit.position in self.snake.body:
            self.fruit.randomize()
        
    def update(self):
        if self.game_won == False:
            self.snake.move_snake()
            self.check_collision()
            self.check_fail()
            self.check_win()
        
    def draw_elements(self):
        self.snake.draw_snake()
        if self.game_won == False:
            self.fruit.draw_fruit()
            self.draw_score()
        else:
            text_surface = game_font.render("You beat Snake", True, pygame.Color('white'))
            text_x = int(cell_size*cell_number/2)
            text_y = int(cell_size*cell_number/2)
            text_rect = text_surface.get_rect(center = (text_x, text_y))
            screen.blit(text_surface, text_rect)
            pygame.time.delay(5000)
            self.game_over()
        
    def check_collision(self):
        if self.fruit.position == self.snake.body[0]:
            self.fruit.randomize()
            self.snake.add_block()
        while self.fruit.position in self.snake.body:
            self.fruit.randomize()
            
    def check_fail(self):
        if not 0 <= self.snake.body[0].x < cell_number or not 0 <= self.snake.body[0].y < cell_number:
            self.game_over()
        for block in self.snake.body[1:]:
            if block == self.snake.body[0]:
                self.game_over()
    
    def check_win(self):
        total_cells = cell_number * cell_number
        if len(self.snake.body) == total_cells:
            self.game_won = True

    def game_over(self):
        self.snake.reset()
        
    def draw_score(self):
        score_text = str(len(self.snake.body) - 3)
        score_surface = game_font.render(score_text, True, pygame.Color('white'))
        score_x = int(cell_size*cell_number - 60)
        score_y = int(cell_size*cell_number - 40)
        score_rect = score_surface.get_rect(center = (score_x, score_y))
        screen.blit(score_surface, score_rect)

class SNAKE:
    def __init__(self):
        self.body = [Vector2(5, 10), Vector2(4, 10), Vector2(3, 10)]
        self.direction = Vector2(0, 0)
        self.new_block = False
        self.can_turn = True
        
    def draw_snake(self):
        for index, block in enumerate(self.body):
            x_position = int(block.x * cell_size) + shrink_amount
            y_position = int(block.y * cell_size) + shrink_amount
            block_width = cell_size - (shrink_amount * 2) 
            block_height = cell_size - (shrink_amount * 2)
            
            snake_rect = pygame.Rect(x_position, y_position, block_width, block_height)
            pygame.draw.rect(screen, (snake_r, snake_g, snake_b), snake_rect)
            
            if index < len(self.body) - 1:
                relation = self.body[index] - self.body[index + 1]
                connection_rect = None
                
                if relation == Vector2(1, 0):
                    connection_rect = pygame.Rect(x_position - (shrink_amount * 2), y_position, shrink_amount * 2, block_height)
                elif relation == Vector2(-1, 0):
                    connection_rect = pygame.Rect(x_position + block_width, y_position, shrink_amount * 2, block_height)
                elif relation == Vector2(0, 1):
                    connection_rect = pygame.Rect(x_position, y_position - (shrink_amount * 2), block_width, shrink_amount * 2)
                elif relation == Vector2(0, -1):
                    connection_rect = pygame.Rect(x_position, y_position + block_height, block_width, shrink_amount * 2)
                if connection_rect:
                    pygame.draw.rect(screen, (snake_r, snake_g, snake_b), connection_rect)
            
    def move_snake(self):
        if self.direction == Vector2(0, 0):
            return

        if self.new_block == True:
            body_copy = self.body[:]
            body_copy.insert(0, body_copy[0] + self.direction)
            self.body = body_copy[:]
            self.new_block = False
        else:
            body_copy = self.body[:-1] # Copies the whole list (body) expept the last part
            body_copy.insert(0, body_copy[0] + self.direction) # Adding element to the front, the element is gonna be the first part of the body plus a player input
            self.body = body_copy[:]
        self.can_turn = True
        
    def add_block(self):
        self.new_block = True
        
    def reset(self):
        self.body = [Vector2(5, 10), Vector2(4, 10), Vector2(3, 10)]
        self.direction = Vector2(0, 0)
            
class FRUIT:
    def __init__(self):
        self.randomize()
        
    def draw_fruit(self):
        offset = int((cell_size - fruit_size)/2)
        x_position = int(self.position.x*cell_size) + offset
        y_position = int(self.position.y*cell_size) + offset
        fruit_width = fruit_size
        fruit_height = fruit_size
        fruit_rect = pygame.Rect(x_position, y_position, fruit_width, fruit_height)
        pygame.draw.rect(screen, (fruit_r, fruit_g, fruit_b), fruit_rect)
        
    def randomize(self):
        self.x = random.randint(0, cell_number-1)
        self.y = random.randint(0, cell_number-1)
        self.position = Vector2(self.x, self.y)

SCREEN_UPDATE = pygame.USEREVENT
pygame.time.set_timer(SCREEN_UPDATE, timer)

main_game = MAIN()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == SCREEN_UPDATE:
            main_game.update()
        if event.type == pygame.KEYDOWN:
            if main_game.snake.can_turn == True:
                if event.key == pygame.K_UP or event.key == pygame.K_w:
                    if main_game.snake.direction.y != 1:
                        main_game.snake.direction = Vector2(0, -1)
                        main_game.snake.can_turn = False
                if event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    if main_game.snake.direction.y != -1:
                        main_game.snake.direction = Vector2(0, 1)
                        main_game.snake.can_turn = False
                if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    if main_game.snake.direction.x != 1:
                        main_game.snake.direction = Vector2(-1, 0)
                        main_game.snake.can_turn = False
                if event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    if main_game.snake.direction.x != -1:
                        main_game.snake.direction = Vector2(1, 0)
                        main_game.snake.can_turn = False
                    
    screen.fill((screen_r, screen_g, screen_b))
    main_game.draw_elements()
    pygame.display.update()
    clock.tick(fps)