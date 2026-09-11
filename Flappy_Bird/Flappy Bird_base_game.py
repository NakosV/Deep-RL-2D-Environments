# Tutorial: https://www.youtube.com/watch?v=GiUGVOqqCKg&list=PLjcN1EyupaQkz5Olxzwvo1OzDNaNLGWoJ
# Assets: https://github.com/russs123/pygame_flappy_bird_assets

import pygame
from pygame.locals import *
import random
import os
import csv

pygame.init()

clock = pygame.time.Clock()
fps = 60

# Game Variables
screen_width = 864
screen_height = 936
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Flappy Bird')
ground_move = 0
speed = 3
is_flying = False
game_over = False
pipe_gap = 150
pipe_frequency = 1650 #miliseconds
last_pipe =  pygame.time.get_ticks() - pipe_frequency
score = 0
pass_pipe = False
font = pygame.font.SysFont('Bauhaus 93', 60)
colour = (255, 255, 255) # white


# Game assets from https://github.com/russs123/pygame_flappy_bird_assets
background = pygame.image.load ("/home/nakos/Desktop/img/bg.png")
ground = pygame.image.load ("/home/nakos/Desktop/img/ground.png")
pipe = pygame.image.load ("/home/nakos/Desktop/img//pipe.png")
button_img = pygame.image.load ("/home/nakos/Desktop/img/restart.png")


def reset_game():
    pipe_group.empty()
    flappy.rect.x = 100
    flappy.rect.y = int(screen_height/2)
    score = 0
    return score
    
def draw_text(text, font, colour, x, y):
    img = font.render(text, True, colour)
    screen.blit(img, (x, y))
    
def save_score_to_csv(current_score):
    filename = "flappy_bird_game_csv.csv"
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
        writer.writerow([f"Try {try_num}", current_score])
        
class Bird(pygame.sprite.Sprite):
    def __init__(self, x, y):
        
        pygame.sprite.Sprite.__init__(self)
        self.images = []
        self.index = 0
        self.counter = 0
        for num in range(1, 4):
            img = pygame.image.load (f"/home/nakos/Desktop/img/bird{num}.png")
            self.images.append(img)
        self.image = self.images[self.index]
        self.rect = self.image.get_rect()
        self.rect.center = [x, y]
        self.velocity = 0
        self.trigger = False
    
    def update(self):
        
        key = pygame.key.get_pressed()
        if is_flying == True:
            # Gravity
            self.velocity = self.velocity + 0.5
            if self.velocity > 8:
                self.velocity = 8
            if self.rect.bottom < 768:    
                self.rect.y = self.rect.y + int(self.velocity)
                
        if game_over == False: 
            # Jumping
            if (pygame.mouse.get_pressed()[0] == 1 or key[pygame.K_SPACE]) and self.trigger == False:
                self.trigger = True
                if self.rect.top > 0:
                    self.velocity = -10
            if pygame.mouse.get_pressed()[0] == 0 and not key[pygame.K_SPACE]:
                self.trigger = False
             
        # Handles the flapping of the bird
            self.counter = self.counter + 1
            flap_cooldwon = 8
            if self.counter > flap_cooldwon:
                self.counter = 0
                self.index = self.index + 1
                if self.index >= len(self.images):
                    self.index = 0
            self.image = self.images[self.index]
            
            self.image = pygame.transform.rotate(self.images[self.index], self.velocity * -2)
        else:
            self.image = pygame.transform.rotate(self.images[self.index], -90)
            
class Pipe(pygame.sprite.Sprite):
    def __init__(self, x, y, position):
        pygame.sprite.Sprite.__init__(self)
        self.image = pipe
        self.rect = self.image.get_rect()
        if position == 1: # 1 for top, -1 for bottom
            self.image = pygame.transform.flip(self.image, False, True) # False is for the x axes and yes for the y axes
            self.rect.bottomleft = [x, y - int(pipe_gap/2)]
        if position == -1:
            self.rect.topleft = [x, y + int(pipe_gap/2)]
    
    def update(self):
        self.rect.x = self.rect.x - speed
        if self.rect.right < 0:
            self.kill()   
            
class Button():
    def __init__(self, x, y, image):
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y)

    def draw(self):
        action = False
        mouse_position = pygame.mouse.get_pos()
        if self.rect.collidepoint(mouse_position):
            if pygame.mouse.get_pressed()[0] == 1:
                action = True
        screen.blit(self.image, (self.rect.x, self.rect.y))
        return action
        
bird_group = pygame.sprite.Group()
pipe_group = pygame.sprite.Group()

flappy = Bird(100, int(screen_height / 2))

bird_group.add(flappy)

button = Button(screen_width / 2 -50, screen_height / 2 - 100, button_img)

run = True
while run:
    clock.tick(fps)
    screen.blit(background, (0,0))
    bird_group.draw(screen)
    bird_group.update()
    pipe_group.draw(screen)
    screen.blit(ground, (ground_move,768))
    
    if len(pipe_group) > 0:
        if bird_group.sprites()[0].rect.left > pipe_group.sprites()[0].rect.left and bird_group.sprites()[0].rect.right < pipe_group.sprites()[0].rect.right and pass_pipe == False:
            pass_pipe = True
        if pass_pipe == True:
            if bird_group.sprites()[0].rect.left > pipe_group.sprites()[0].rect.right:
                score = score + 1
                pass_pipe = False
    
    draw_text(f"Score: {score}", font, colour, int(screen_width/2)-80, 20)
    
    if pygame.sprite.groupcollide(bird_group, pipe_group, False, False):
        if game_over == False:
            save_score_to_csv(score)
        game_over = True

    if flappy.rect.bottom >= 768:
        if game_over == False:
            save_score_to_csv(score)
        game_over = True
        is_flying = False
    
    if game_over == False and is_flying == True:
        time_now = pygame.time.get_ticks()
        pipe_height = random.randint(-150, 150)
        if time_now - last_pipe > pipe_frequency:
            bottom_pipe = Pipe(screen_width, int(screen_height/2) + pipe_height, -1)
            top_pipe = Pipe(screen_width, int(screen_height/2) + pipe_height, 1)
            pipe_group.add(bottom_pipe)
            pipe_group.add(top_pipe)
            last_pipe = time_now
            
        ground_move = ground_move - speed
        if abs(ground_move) > 35:
            ground_move = 0
            
        pipe_group.update()
            
    if game_over == True:
        if button.draw() == True:
            game_over = False
            score = reset_game()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if (event.type == pygame.MOUSEBUTTONDOWN or (event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE)) and is_flying == False and game_over == False:
            is_flying = True
    pygame.display.update()
            
pygame.quit()