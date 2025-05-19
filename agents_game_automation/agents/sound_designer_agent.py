# agents/sound_designer_agent.py

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import torch
import os
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils.constants as Constants

class SoundDesignerAgent:
    def __init__(self, configPath):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[LOG] Using device: {self.device}")
        print("[LOG] Loading MusicGen model...")
        self.model = MusicGen.get_pretrained('melody')
        print("[LOG] Model loaded successfully!")
        self.configPath = configPath

    def generate_sounds(self):
        print(f"🎵 Generating sound assets from game config")
        output_dir = "assets/sounds"
        os.makedirs(output_dir, exist_ok=True)

        # Load config
        with open(self.configPath, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Generate prompts from sounds_required
        tasks = []

        # Start-screen music
        start_music = config.get("game_info", {}) \
                            .get("start_screen", {}) \
                            .get("music", {})
        if start_music:
            prompt = f"High-quality sound design for game audio: {start_music.get('description', '')}"
            path = start_music.get("assert_path", "")
            duration = start_music.get("duration", 3)
            tasks.append((prompt, path, duration))

        # Game-screen music
        game_music = config.get("screens", {}) \
                        .get("game_screen", {}) \
                        .get("music", {})
        if game_music:
            prompt = f"High-quality sound design for game audio: {game_music.get('description', '')}"
            path = game_music.get("assert_path", "")
            duration = game_music.get("duration", 3)
            tasks.append((prompt, path, duration))

        # Win/lose screen sounds
        for key in ("win_screen", "lose_screen"):
            sound_info = config.get("screens", {}).get(key, {}).get("sound", {})
            if sound_info:
                prompt = f"High-quality sound design for game audio: {sound_info.get('description', '')}"
                path = sound_info.get("assert_path", "")
                duration = sound_info.get("duration", 3)
                tasks.append((prompt, path, duration))

        # Gameplay rule sounds
        for rule in config.get("gameplay", {}).get("rules", []):
            sound_path = rule.get("assert_path", "")
            desc = rule.get("effect", rule.get("on", ""))
            if sound_path:
                prompt = f"High-quality sound design for game audio: {desc}"
                duration = rule.get("duration", 3)
                tasks.append((prompt, sound_path, duration))


        for desc, filename, duration_sec in tasks:
            print(f"🎧 Generating: {desc}")
            self.model.set_generation_params(duration=duration_sec)
            wavs = self.model.generate([desc])
            sound_path = filename.replace(".wav","")
            audio_write(sound_path, wavs[0].cpu(), sample_rate=32000)
            print(f"[SAVED] {sound_path}")

        print(f"✅ {len(tasks)} sound assets saved to {output_dir}")

if __name__ == "__main__":
    agent = SoundDesignerAgent(Constants.GAME_CONFIGV3)
    agent.generate_sounds()
