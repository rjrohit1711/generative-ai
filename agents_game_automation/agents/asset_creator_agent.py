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

        output_dir = "outputs/assets"
        os.makedirs(output_dir, exist_ok=True)

        # Track filename-safe prompt labels
        tasks = []

        # 3. Assets
        for asset_type, asset_list in config.get("assets_required", {}).items():
            for asset in asset_list:
                prompt = f"Give image that I can use in game development for assert type: {asset_type} asset: {asset}"
                filename = f"{asset_type.lower()}_{asset.replace(' ', '_').lower()}.png"
                tasks.append((prompt, filename))

        # Generate and save each image
        for prompt, filename in tasks:
            print(f"🖼️  Generating: {prompt}")
            image = self.pipe(prompt).images[0]
            image_path = os.path.join(output_dir, filename)
            image.save(image_path)

        print(f"✅ {len(tasks)} assets saved to {output_dir}")
   
if __name__ == "__main__":
    agent = AssetCreatorAgent()
    agent.generate_assets()