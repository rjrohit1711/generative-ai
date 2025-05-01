# agents/asset_creator_agent.py

from diffusers import StableDiffusionPipeline
import torch
import os
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils.constants as Constants

class AssetCreatorAgent:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

    def generate_assets(self):
        print("🎨 Generating visual assets...")

        # Load game configuration
        with open(Constants.GAME_CONFIG, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Track filename-safe prompt labels
        tasks = []

        # 3. Assets
        for asset_type, asset_list in config.get("assets_required", {}).items():
            for asset in asset_list:
                prompt = (
                    f"Ultra-detailed, high-resolution concept art for a 2d mobile Game {asset_type}: {asset}. "
                    "4K, sharp focus, 2D painting style, "
                )
                filename = f"{asset.replace(' ', '_').lower()}.png"
                tasks.append((prompt, filename, asset_type))

        # Generate and save each image
        for prompt, filename, asset_type in tasks:
            print(f"🖼️  Generating: {prompt}")
            image = self.pipe(prompt,  num_inference_steps=150, guidance_scale=8.5).images[0]
            output_dir = f"bin/source/assets/{asset_type}/"
            os.makedirs(output_dir, exist_ok=True)
            image_path = os.path.join(output_dir, filename)
            image.save(image_path)

        print(f"✅ {len(tasks)} assets saved to {output_dir}")
   
if __name__ == "__main__":
    agent = AssetCreatorAgent()
    agent.generate_assets()