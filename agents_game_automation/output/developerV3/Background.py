"""
This module provides a Background class for creating a scrolling background in a Pygame application.
The background image scrolls vertically at a specified speed, creating a continuous scrolling effect.
"""

import pygame
import time

SCENE_ASSET = "./assets/sprites/scene_background.png"
SCREEN_WIDTH = 500
SCREEN_HEIGHT = 600
SCROLL_SPEED = 4

class Background:
    def __init__(self, screen_width, screen_height, scroll_speed):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.scroll_speed = scroll_speed
        self.background_image = pygame.image.load(SCENE_ASSET)
        self.background_rect = self.background_image.get_rect()
        self.y_offset = 0

    def update(self):
        self.y_offset += self.scroll_speed
        if self.y_offset >= self.screen_height:
            self.y_offset = 0

    def draw(self, screen):
        screen.blit(self.background_image, (0, self.y_offset - self.screen_height))
        screen.blit(self.background_image, (0, self.y_offset))

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Background Test")
    background = Background(SCREEN_WIDTH, SCREEN_HEIGHT, SCROLL_SPEED)
    start_time = time.time()
    running = True
    while running:
        if time.time() - start_time > 10:
            running = False
        background.update()
        background.draw(screen)
        pygame.display.flip()
        pygame.time.Clock().tick(60)
    pygame.quit()