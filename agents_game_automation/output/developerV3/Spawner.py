"""
Spawner Module

This module defines the `Spawner` class, which is responsible for spawning, updating, drawing, and managing entities
such as collectibles and obstacles in a game. The spawner controls the frequency of entity spawning, the maximum number
of entities allowed on the screen at once, and their fall speed. It also handles collision detection between entities
and a player rectangle.
"""

import pygame
import time
import random
from Apple import Apple
from Stone import Stone

class Spawner:
    def __init__(self, spawner_id, entity_type, asset, interval_sec, max_on_screen, fall_speed):
        self.spawner_id = spawner_id
        self.entity_type = entity_type
        self.asset = asset
        self.interval_sec = interval_sec
        self.max_on_screen = max_on_screen
        self.fall_speed = fall_speed
        self.last_spawn_time = time.time()
        self.entities = []

    def spawn_entity(self, screen_width):
        current_time = time.time()
        if current_time - self.last_spawn_time > self.interval_sec and len(self.entities) < self.max_on_screen:
            x_position = random.randint(0, screen_width - 50)  # Assuming entities are 50x50 pixels
            if self.entity_type == "collectible":
                entity = Apple(screen_width, 600, x_position, self.fall_speed)
            elif self.entity_type == "obstacle":
                entity = Stone(screen_width, 600, x_position)  # Removed fall_speed argument here
            self.entities.append(entity)
            self.last_spawn_time = current_time

    def update_entities(self):
        for entity in self.entities:
            entity.update()
            if entity.rect.top > 600:  # Remove entities that fall off the screen
                self.remove_entity(entity)

    def draw_entities(self, screen):
        for entity in self.entities:
            entity.draw(screen)

    def check_collisions(self, player_rect):
        collided_entities = []
        for entity in self.entities:
            if entity.check_collision(player_rect):
                collided_entities.append(entity)
        return collided_entities

    def remove_entity(self, entity):
        self.entities.remove(entity)

if __name__ == "__main__":
    pygame.init()
    screen_width, screen_height = 800, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Spawner Test")

    spawner = Spawner(spawner_id=2, entity_type="obstacle", asset=None, interval_sec=1, max_on_screen=5, fall_speed=5)

    start_time = time.time()
    running = True
    while running and time.time() - start_time < 10:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))

        spawner.spawn_entity(screen_width)
        spawner.update_entities()
        spawner.draw_entities(screen)

        player_rect = pygame.Rect(400, 550, 50, 50)
        collided_entities = spawner.check_collisions(player_rect)
        for entity in collided_entities:
            print(f"Collision detected with entity at {entity.rect.topleft}")

        pygame.display.flip()
        pygame.time.Clock().tick(60)

    pygame.quit()