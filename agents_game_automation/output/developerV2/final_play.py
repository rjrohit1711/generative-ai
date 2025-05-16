"""
play.py

This module contains the main game logic for the "Apple Hop" game. It initializes the game window, handles player input,
manages game entities (player, apples, and stones), checks for collisions, updates the game state, and displays the game
HUD (heads-up display) including the score and health. The game includes start, win, and lose screens, and plays background
music and sound effects for various game events.
"""

import pygame
import sys
import os
import random

# Constants
GAME_NAME = "Apple Hop"
GAME_VERSION = "1.0.0"
SCREEN_WIDTH = 500
SCREEN_HEIGHT = 600
FRAME_RATE = 60
SCROLL_SPEED = 4

PLAYER_ID = "player"
APPLE_ID = "apple"
STONE_ID = "stone"

APPLE_SPAWNER_ID = "apple_spawner"
STONE_SPAWNER_ID = "stone_spawner"

APPLE_SPAWN_INTERVAL = 1.5
STONE_SPAWN_INTERVAL = 1
APPLE_FALL_SPEED = 2
STONE_FALL_SPEED = 1
APPLE_MAX_ON_SCREEN = 4
STONE_MAX_ON_SCREEN = 4

TARGET_SCORE = 15

BACKGROUND_PATH = "assets/sprites/scene_background.png"
PLAYER_SPRITE_PATH = "assets/sprites/bouncey.png"
APPLE_SPRITE_PATH = "assets/sprites/apple.png"
STONE_SPRITE_PATH = "assets/sprites/stone.png"

BACKGROUND_MUSIC_PATH = "assets/sounds/background_music.wav"
APPLE_CATCH_SOUND_PATH = "assets/sounds/apple_catch.wav"
STONE_HIT_SOUND_PATH = "assets/sounds/stone_hit.wav"
WIN_SOUND_PATH = "assets/sounds/win.wav"
LOSE_SOUND_PATH = "assets/sounds/lose.wav"

SCORE_FONT_SIZE = 24
HEALTH_FONT_SIZE = 24
SCORE_COLOR = (255, 255, 255)
HEALTH_COLOR = (255, 0, 0)
SCORE_POSITION = (10, 10)
HEALTH_POSITION = (10, 40)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption(GAME_NAME)
clock = pygame.time.Clock()

# Load assets
background = pygame.image.load(BACKGROUND_PATH).convert()
player_sprite = pygame.image.load(PLAYER_SPRITE_PATH).convert_alpha()
apple_sprite = pygame.image.load(APPLE_SPRITE_PATH).convert_alpha()
stone_sprite = pygame.image.load(STONE_SPRITE_PATH).convert_alpha()

# Load sounds
background_music = pygame.mixer.Sound(BACKGROUND_MUSIC_PATH)
apple_catch_sound = pygame.mixer.Sound(APPLE_CATCH_SOUND_PATH)
stone_hit_sound = pygame.mixer.Sound(STONE_HIT_SOUND_PATH)
win_sound = pygame.mixer.Sound(WIN_SOUND_PATH)
lose_sound = pygame.mixer.Sound(LOSE_SOUND_PATH)

# Fonts
score_font = pygame.font.Font(None, SCORE_FONT_SIZE)
health_font = pygame.font.Font(None, HEALTH_FONT_SIZE)


class Entity(pygame.sprite.Sprite):
    def __init__(self, image_path, x, y):
        super().__init__()
        self.image = pygame.image.load(image_path).convert_alpha()
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y


class Player(Entity):
    def __init__(self):
        super().__init__(PLAYER_SPRITE_PATH, SCREEN_WIDTH // 2, SCREEN_HEIGHT - 100)
        self.speed = 5

    def update(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and self.rect.left > 0:
            self.rect.x -= self.speed
        if keys[pygame.K_RIGHT] and self.rect.right < SCREEN_WIDTH:
            self.rect.x += self.speed


def load_and_play_background_music():
    pygame.mixer.music.load(BACKGROUND_MUSIC_PATH)
    pygame.mixer.music.play(-1)


def show_start_screen():
    start_font = pygame.font.Font(None, 48)
    text = start_font.render("Press SPACE to Start", True, (255, 255, 255))
    text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
    screen.blit(text, text_rect)
    pygame.display.flip()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    waiting = False


def show_win_screen():
    win_font = pygame.font.Font(None, 48)
    text = win_font.render("You Win! Press SPACE to Restart", True, (255, 255, 255))
    text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
    screen.blit(text, text_rect)
    pygame.display.flip()
    win_sound.play()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    waiting = False
                    main()


def show_lose_screen():
    lose_font = pygame.font.Font(None, 48)
    text = lose_font.render("You Lose! Press SPACE to Restart", True, (255, 255, 255))
    text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
    screen.blit(text, text_rect)
    pygame.display.flip()
    lose_sound.play()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    waiting = False
                    main()


def spawn_entity(entity_id, entities):
    if entity_id == APPLE_ID and len([e for e in entities if e.id == APPLE_ID]) < APPLE_MAX_ON_SCREEN:
        x = random.randint(0, SCREEN_WIDTH - apple_sprite.get_width())
        y = -apple_sprite.get_height()
        apple = Entity(APPLE_SPRITE_PATH, x, y)
        apple.id = APPLE_ID
        entities.add(apple)
    elif entity_id == STONE_ID and len([e for e in entities if e.id == STONE_ID]) < STONE_MAX_ON_SCREEN:
        x = random.randint(0, SCREEN_WIDTH - stone_sprite.get_width())
        y = -stone_sprite.get_height()
        stone = Entity(STONE_SPRITE_PATH, x, y)
        stone.id = STONE_ID
        entities.add(stone)


def handle_collisions(player, entities, score, health):
    for entity in entities:
        if pygame.sprite.collide_rect(player, entity):
            if entity.id == APPLE_ID:
                score += 1
                apple_catch_sound.play()
            elif entity.id == STONE_ID:
                health -= 1
                stone_hit_sound.play()
            entities.remove(entity)
    return score, health


def update_entities(entities):
    for entity in entities:
        if entity.id == APPLE_ID:
            entity.rect.y += APPLE_FALL_SPEED
        elif entity.id == STONE_ID:
            entity.rect.y += STONE_FALL_SPEED
        if entity.rect.y > SCREEN_HEIGHT:
            entities.remove(entity)


def draw_entities(screen, entities):
    for entity in entities:
        screen.blit(entity.image, entity.rect)


def draw_hud(screen, score, health):
    score_text = score_font.render(f"Score: {score}", True, SCORE_COLOR)
    health_text = health_font.render(f"Health: {health}", True, HEALTH_COLOR)
    screen.blit(score_text, SCORE_POSITION)
    screen.blit(health_text, HEALTH_POSITION)


def main():
    load_and_play_background_music()
    show_start_screen()

    player = Player()
    entities = pygame.sprite.Group()
    last_apple_spawn_time = 0
    last_stone_spawn_time = 0
    score = 0
    health = 3

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        current_time = pygame.time.get_ticks()

        if current_time - last_apple_spawn_time > APPLE_SPAWN_INTERVAL * 1000:
            spawn_entity(APPLE_ID, entities)
            last_apple_spawn_time = current_time

        if current_time - last_stone_spawn_time > STONE_SPAWN_INTERVAL * 1000:
            spawn_entity(STONE_ID, entities)
            last_stone_spawn_time = current_time

        player.update()
        update_entities(entities)
        score, health = handle_collisions(player, entities, score, health)

        screen.blit(background, (0, 0))
        draw_entities(screen, entities)
        draw_hud(screen, score, health)
        screen.blit(player.image, player.rect)
        pygame.display.flip()

        if health <= 0:
            show_lose_screen()
            running = False
        if score >= TARGET_SCORE:
            show_win_screen()
            running = False

        clock.tick(FRAME_RATE)

    pygame.quit()


if __name__ == "__main__":
    main()