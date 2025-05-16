"""
Game.py

This module defines the main game logic for the "Apple Hop" game. It initializes the game window, manages game objects such as the player, apples, stones, background, and handles user input, updates game state, and renders the game screen. The game includes a scoring system and health management to determine win or loss conditions.
"""

import pygame
from Player import Player
from Apple import Apple
from Stone import Stone
from Spawner import Spawner
from Background import Background
from HUD import HUD
from SoundManager import SoundManager

GAME_NAME = "Apple Hop"
GAME_VERSION = "1.0.0"
GAME_DESCRIPTION = "Bouncey moves left and right to catch falling apples while avoiding stones!"

SCREEN_WIDTH = 500
SCREEN_HEIGHT = 600
FRAME_RATE = 60
SCROLL_SPEED = 4

LEVEL_1_ID = "level_1"
LEVEL_1_SPAWNERS = ["apple_spawner", "stone_spawner"]

GAMEPLAY_TYPE = "score"
GAMEPLAY_TARGET = 15
GAMEPLAY_DESCRIPTION = "Catch apples to earn points. Avoid stones or lose health."
GAMEPLAY_WIN_MESSAGE = "You collected 15 apples! Level Complete!"
GAMEPLAY_LOSE_MESSAGE = "Bouncey ran out of health! Game Over!"

APPLE_CATCH_ID = "apple_catch"
STONE_HIT_ID = "stone_hit"
WIN_ID = "win"
LOSE_ID = "lose"


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(GAME_NAME)
        self.clock = pygame.time.Clock()
        self.running = True
        self.score = 0
        self.health = 3
        self.level = LEVEL_1_ID
        self.player = Player(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.background = Background(SCREEN_WIDTH, SCREEN_HEIGHT, SCROLL_SPEED)
        self.hud = HUD(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.sound_manager = SoundManager()
        self.spawners = self.initialize_spawners()
        self.show_start_screen()

    def initialize_spawners(self):
        spawners = {}
        for spawner_id in LEVEL_1_SPAWNERS:
            if spawner_id == "apple_spawner":
                spawners[spawner_id] = Spawner(spawner_id, "collectible", None, 1.5, 4, 2)
            elif spawner_id == "stone_spawner":
                spawners[spawner_id] = Spawner(spawner_id, "obstacle", None, 1, 4, 1)
        return spawners

    def handle_input(self, keys):
        self.player.handle_input(keys)

    def update(self):
        self.player.update()
        self.background.update()
        self.hud.update_score(self.score)
        self.hud.update_health(self.health)
        for spawner in self.spawners.values():
            spawner.spawn_entity(SCREEN_WIDTH)
            spawner.update_entities()
            collided_entities = spawner.check_collisions(self.player.rect)
            for entity in collided_entities:
                if entity.entity_type == "collectible":
                    self.score += 1
                    self.sound_manager.play_sound_effect(APPLE_CATCH_ID)
                    spawner.remove_entity(entity)
                elif entity.entity_type == "obstacle":
                    self.health -= 1
                    self.sound_manager.play_sound_effect(STONE_HIT_ID)
                    spawner.remove_entity(entity)
        if self.score >= GAMEPLAY_TARGET:
            self.win_game()
        if self.health <= 0:
            self.lose_game()

    def draw(self):
        self.background.draw(self.screen)
        self.player.draw(self.screen)
        for spawner in self.spawners.values():
            spawner.draw_entities(self.screen)
        self.hud.draw(self.screen)
        pygame.display.flip()

    def win_game(self):
        self.sound_manager.play_sound_effect(WIN_ID)
        self.show_win_screen()
        self.running = False

    def lose_game(self):
        self.sound_manager.play_sound_effect(LOSE_ID)
        self.show_lose_screen()
        self.running = False

    def run(self):
        self.sound_manager.play_background_music()
        while self.running:
            keys = pygame.key.get_pressed()
            self.handle_input(keys)
            self.update()
            self.draw()
            self.clock.tick(FRAME_RATE)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
        self.sound_manager.stop_background_music()
        pygame.quit()

    def show_start_screen(self):
        self.screen.fill((0, 0, 0))
        font = pygame.font.Font(None, 36)
        text = font.render("Press any key to start", True, (255, 255, 255))
        text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        self.screen.blit(text, text_rect)
        pygame.display.flip()
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    waiting = False
                elif event.type == pygame.KEYUP:
                    waiting = False

    def show_win_screen(self):
        self.screen.fill((0, 200, 0))
        font = pygame.font.Font(None, 36)
        text = font.render(GAMEPLAY_WIN_MESSAGE, True, (255, 255, 255))
        text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        self.screen.blit(text, text_rect)
        pygame.display.flip()
        pygame.time.wait(3000)

    def show_lose_screen(self):
        self.screen.fill((200, 0, 0))
        font = pygame.font.Font(None, 36)
        text = font.render(GAMEPLAY_LOSE_MESSAGE, True, (255, 255, 255))
        text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        self.screen.blit(text, text_rect)
        pygame.display.flip()
        pygame.time.wait(3000)


if __name__ == "__main__":
    game = Game()
    game.run()