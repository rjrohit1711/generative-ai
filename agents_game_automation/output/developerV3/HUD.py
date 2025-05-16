"""
Module: HUD

This module defines a Head-Up Display (HUD) class for a pygame application.
The HUD is responsible for displaying the player's score and health on the screen.
It provides methods to update the score and health values and to draw these values
onto the game screen at specified positions.
"""

import pygame
import time

HUD_SCORE_POSITION_X = 10
HUD_SCORE_POSITION_Y = 10
HUD_SCORE_FONT_SIZE = 24
HUD_SCORE_COLOR = "#FFFFFF"
HUD_SCORE_LABEL = "Score"

HUD_HEALTH_POSITION_X = 420  # Adjusted to ensure health bar is within screen boundaries
HUD_HEALTH_POSITION_Y = 10
HUD_HEALTH_FONT_SIZE = 24
HUD_HEALTH_COLOR = "#FF0000"
HUD_HEALTH_LABEL = "Health"

SCREEN_WIDTH = 500
SCREEN_HEIGHT = 600

class HUD:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.score_font = pygame.font.Font(None, HUD_SCORE_FONT_SIZE)
        self.health_font = pygame.font.Font(None, HUD_HEALTH_FONT_SIZE)
        self.score = 0
        self.health = 3

    def update_score(self, score):
        self.score = score

    def update_health(self, health):
        self.health = health

    def draw(self, screen):
        score_text = self.score_font.render(f"{HUD_SCORE_LABEL}: {self.score}", True, HUD_SCORE_COLOR)
        health_text = self.health_font.render(f"{HUD_HEALTH_LABEL}: {self.health}", True, HUD_HEALTH_COLOR)
        screen.blit(score_text, (HUD_SCORE_POSITION_X, HUD_SCORE_POSITION_Y))
        screen.blit(health_text, (HUD_HEALTH_POSITION_X, HUD_HEALTH_POSITION_Y))

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("HUD Test")
    hud = HUD(SCREEN_WIDTH, SCREEN_HEIGHT)
    hud.update_score(1000)
    hud.update_health(5)
    clock = pygame.time.Clock()
    start_time = time.time()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if time.time() - start_time > 10:
            running = False
        screen.fill((0, 0, 0))
        hud.draw(screen)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()