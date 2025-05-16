# Player.py

"""
Module for the Player class in a pygame-based game. This module defines the Player class,
which handles the player character's attributes, movement, drawing, collision detection,
and health management.
"""

import pygame
import time

PLAYER_ID = "player"
PLAYER_TYPE = "character"
PLAYER_SPRITE = "bouncey"
PLAYER_HEALTH = 3
PLAYER_CONTROLS_LEFT = pygame.K_LEFT
PLAYER_CONTROLS_RIGHT = pygame.K_RIGHT
PLAYER_DESCRIPTION = "Sprite image of the player character Bouncey."
PLAYER_ASSET = "./assets/sprites/bouncey.png"

class Player:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.health = PLAYER_HEALTH
        self.position = [screen_width // 2, screen_height - 50]
        self.sprite = pygame.image.load(PLAYER_ASSET)
        self.rect = self.sprite.get_rect(topleft=self.position)
        self.controls = {PLAYER_CONTROLS_LEFT: False, PLAYER_CONTROLS_RIGHT: False}
        self.speed = 3  # Reduced speed

    def handle_input(self, keys):
        self.controls[PLAYER_CONTROLS_LEFT] = keys[PLAYER_CONTROLS_LEFT]
        self.controls[PLAYER_CONTROLS_RIGHT] = keys[PLAYER_CONTROLS_RIGHT]

    def update(self):
        if self.controls[PLAYER_CONTROLS_LEFT] and self.rect.left > 0:
            self.rect.x -= self.speed
        if self.controls[PLAYER_CONTROLS_RIGHT] and self.rect.right < self.screen_width:
            self.rect.x += self.speed

    def draw(self, screen):
        screen.blit(self.sprite, self.rect.topleft)

    def check_collision(self, entity):
        return self.rect.colliderect(entity.rect)

    def take_damage(self, damage):
        self.health -= damage
        if self.health < 0:
            self.health = 0

if __name__ == "__main__":
    pygame.init()
    
    screen_width, screen_height = 800, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Player Test")

    player = Player(screen_width, screen_height)

    running = True
    while running:
        keys = pygame.key.get_pressed()
        player.handle_input(keys)
        player.update()

        screen.fill((0, 0, 0))
        player.draw(screen)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()