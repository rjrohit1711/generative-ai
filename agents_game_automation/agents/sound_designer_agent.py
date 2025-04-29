# agents/sound_designer_agent.py

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import torch
import os

class SoundDesignerAgent:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[LOG] Using device: {self.device}")
        print("[LOG] Loading MusicGen model...")
        self.model = MusicGen.get_pretrained('melody')
        print("[LOG] Model loaded successfully!")

    def generate_sounds(self, game_description: str):
        print(f"🎵 Generating sound assets for: '{game_description}'")
        output_dir = "outputs/sounds"
        os.makedirs(output_dir, exist_ok=True)

        descriptions = [f"Background music for a game: {game_description}"]
        wavs = self.model.generate(descriptions)

        sound_path = os.path.join(output_dir, "generated_music.wav")
        print(f"[LOG] Saving sound to: {sound_path}")
        audio_write(sound_path, wavs[0].cpu(), sample_rate=32000)

        return [sound_path]
