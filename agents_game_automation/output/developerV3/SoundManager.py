"""
SoundManager Module

This module provides a SoundManager class that handles the playback of background music and sound effects
using the Pygame library. It initializes the necessary Pygame mixer components and manages sound assets
for different game events such as catching apples, hitting stones, winning, and losing.
"""

import pygame
import time

BACKGROUND_MUSIC_ID = "background_music"
BACKGROUND_MUSIC_ASSET = "./assets/sounds/background_music.wav"

APPLE_CATCH_ID = "apple_catch"
APPLE_CATCH_ASSET = "./assets/sounds/apple_catch.wav"

STONE_HIT_ID = "stone_hit"
STONE_HIT_ASSET = "./assets/sounds/stone_hit.wav"

WIN_ID = "win"
WIN_ASSET = "./assets/sounds/win.wav"

LOSE_ID = "lose"
LOSE_ASSET = "./assets/sounds/lose.wav"

class SoundManager:
    def __init__(self):
        pygame.mixer.init()
        self.sounds = {
            BACKGROUND_MUSIC_ID: pygame.mixer.Sound(BACKGROUND_MUSIC_ASSET),
            APPLE_CATCH_ID: pygame.mixer.Sound(APPLE_CATCH_ASSET),
            STONE_HIT_ID: pygame.mixer.Sound(STONE_HIT_ASSET),
            WIN_ID: pygame.mixer.Sound(WIN_ASSET),
            LOSE_ID: pygame.mixer.Sound(LOSE_ASSET)
        }

    def play_background_music(self):
        pygame.mixer.music.load(BACKGROUND_MUSIC_ASSET)
        pygame.mixer.music.play(-1)

    def stop_background_music(self):
        pygame.mixer.music.stop()

    def play_sound_effect(self, sound_id):
        if sound_id in self.sounds:
            self.sounds[sound_id].play()

    def stop_sound_effect(self, sound_id):
        if sound_id in self.sounds:
            self.sounds[sound_id].stop()

if __name__ == "__main__":
    sound_manager = SoundManager()

    print("Playing background music...")
    sound_manager.play_background_music()
    time.sleep(3)

    print("Stopping background music...")
    sound_manager.stop_background_music()
    time.sleep(1)

    print("Playing sound effects...")
    sound_manager.play_sound_effect(APPLE_CATCH_ID)
    time.sleep(1)

    sound_manager.play_sound_effect(STONE_HIT_ID)
    time.sleep(1)

    sound_manager.play_sound_effect(WIN_ID)
    time.sleep(1)

    sound_manager.play_sound_effect(LOSE_ID)
    time.sleep(1)

    print("Stopping sound effects...")
    sound_manager.stop_sound_effect(APPLE_CATCH_ID)
    sound_manager.stop_sound_effect(STONE_HIT_ID)
    sound_manager.stop_sound_effect(WIN_ID)
    sound_manager.stop_sound_effect(LOSE_ID)
    time.sleep(1)

    print("Test completed. Exiting in 1 second...")
    time.sleep(1)