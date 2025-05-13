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
        output_dir = "bin/source/assets/sounds"
        os.makedirs(output_dir, exist_ok=True)

        # Load config
        with open(self.configPath, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Generate prompts from sounds_required
        tasks = []
        for sound_info in config.get("sounds_required", []):
            sound_name = sound_info.get("asset", "")
            description = sound_info.get("description", "")
            duration = sound_info.get("duration", 3)  # default to 3 seconds if not specified
            prompt = (
                f"High-quality sound design for game audio: {sound_name}. {description}"
            )
            filename = f"{sound_name.replace(' ', '_').lower()}"
            tasks.append((prompt, filename, duration))

        for desc, filename, duration_sec in tasks:
            print(f"🎧 Generating: {desc}")
            self.model.set_generation_params(duration=duration_sec)
            wavs = self.model.generate([desc])
            sound_path = os.path.join(output_dir, filename)
            audio_write(sound_path, wavs[0].cpu(), sample_rate=32000)
            print(f"[SAVED] {sound_path}")

        print(f"✅ {len(tasks)} sound assets saved to {output_dir}")

if __name__ == "__main__":
    agent = SoundDesignerAgent(Constants.GAME_CONFIGV2)
    agent.generate_sounds()
