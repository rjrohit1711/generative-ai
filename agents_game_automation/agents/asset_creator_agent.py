# agents/asset_creator_agent.py

from diffusers import StableDiffusionPipeline
import torch
import os

class AssetCreatorAgent:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

    def generate_assets(self, game_description):
        print("🎨 Generating visual assets...")
        prompt = f"Create a game environment based on: {game_description}"
        output_dir = "outputs/assets"
        os.makedirs(output_dir, exist_ok=True)
        image = self.pipe(prompt).images[0]
        image_path = os.path.join(output_dir, "generated_asset.png")
        image.save(image_path)
        return [image_path]
