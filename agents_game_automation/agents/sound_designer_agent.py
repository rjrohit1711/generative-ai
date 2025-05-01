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
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[LOG] Using device: {self.device}")
        print("[LOG] Loading MusicGen model...")
        self.model = MusicGen.get_pretrained('melody')
        print("[LOG] Model loaded successfully!")

    def generate_sounds(self):
        print(f"🎵 Generating sound assets from game config")
        output_dir = "bin/source/assets/sounds"
        os.makedirs(output_dir, exist_ok=True)

        # Load config
        with open(Constants.GAME_CONFIG, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Generate prompts from sounds_required
        tasks = []
        for sound_type, sound_list in config.get("sounds_required", {}).items():
            for sound in sound_list:
                description = f"{sound_type} sound for game: {sound}"
                filename = f"{sound.replace(' ', '_').lower()}"
                tasks.append((description, filename))

        for desc, filename in tasks:
            print(f"🎧 Generating: {desc}")
            wavs = self.model.generate([desc])
            sound_path = os.path.join(output_dir, filename)
            audio_write(sound_path, wavs[0].cpu(), sample_rate=32000)
            print(f"[SAVED] {sound_path}")

        print(f"✅ {len(tasks)} sound assets saved to {output_dir}")

if __name__ == "__main__":
    agent = SoundDesignerAgent()
    agent.generate_sounds()
