"""
This module defines the Apple class, which represents an apple collectible in a pygame-based game.
The apple falls from the top of the screen and can collide with the player. It includes methods for
updating its position, drawing it on the screen, and checking for collisions with the player.
"""

import pygame
import time

APPLE_ID = "apple"
APPLE_TYPE = "collectible"
APPLE_SPRITE = "apple"
APPLE_SCORE = 1
APPLE_DESCRIPTION = "Sprite image for the collectible apple."
APPLE_ASSET = "./assets/sprites/apple.png"
APPLE_FALL_SPEED = 2

class Apple:
    def __init__(self, screen_width, screen_height, x_position, y_position):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.position = [x_position, y_position]
        self.sprite = pygame.image.load(APPLE_ASSET)
        self.rect = self.sprite.get_rect(topleft=self.position)
        self.fall_speed = APPLE_FALL_SPEED
        self.entity_type = APPLE_TYPE

    def update(self):
        self.position[1] += self.fall_speed
        self.rect.topleft = self.position

    def draw(self, screen):
        screen.blit(self.sprite, self.rect.topleft)

    def check_collision(self, player_rect):
        return self.rect.colliderect(player_rect)

def test_apple():
    pygame.init()
    screen_width, screen_height = 800, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Apple Test")

    apple = Apple(screen_width, screen_height, 100, 0)
    player_rect = pygame.Rect(100, 550, 50, 50)

    apple.update()
    apple.draw(screen)

    collision = apple.check_collision(player_rect)
    print(f"Collision detected: {collision}")

    pygame.display.flip()
    time.sleep(10)
    pygame.quit()

if __name__ == "__main__":
    test_apple()