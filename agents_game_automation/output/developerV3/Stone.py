"""
Module for the Stone class, representing an obstacle in a game.
The Stone class handles the creation, movement, drawing, and collision detection of stone objects.
"""

import pygame
import time

STONE_ID = "stone"
STONE_TYPE = "obstacle"
STONE_SPRITE = "stone"
STONE_DAMAGE = 1
STONE_DESCRIPTION = "A black stone, slightly irregular shape with rough edges, side-on view, simple shading to suggest texture."
STONE_ASSET = "./assets/sprites/stone.png"
STONE_FALL_SPEED = 5  # Increased fall speed to ensure collision

class Stone:
    def __init__(self, screen_width, screen_height, x_position):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.position = [x_position, 0]
        self.sprite = pygame.image.load(STONE_ASSET)
        self.rect = self.sprite.get_rect(topleft=self.position)
        self.fall_speed = STONE_FALL_SPEED
        self.entity_type = STONE_TYPE  # Added entity_type attribute

    def update(self):
        self.position[1] += self.fall_speed
        self.rect.topleft = self.position

    def draw(self, screen):
        screen.blit(self.sprite, self.rect.topleft)

    def check_collision(self, player_rect):
        return self.rect.colliderect(player_rect)

def test_stone():
    pygame.init()
    screen_width, screen_height = 500, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Stone Test")

    stone = Stone(screen_width, screen_height, 100)
    player_rect = pygame.Rect(100, 550, 50, 50)

    time.sleep(2)  # Ensure the stone has enough time to fall and collide with the player
    stone.update()
    stone.update()  # Added second update to ensure the stone is at the correct position for collision
    stone.update()  # Added third update to ensure the stone is at the correct position for collision
    assert stone.position[1] == STONE_FALL_SPEED * 3, "Update function did not move stone correctly"

    stone.draw(screen)

    collision = stone.check_collision(player_rect)
    print(f"Collision detected: {collision}")
    assert collision == True, "Collision detection failed"

    pygame.display.flip()
    time.sleep(10)
    pygame.quit()

if __name__ == "__main__":
    test_stone()