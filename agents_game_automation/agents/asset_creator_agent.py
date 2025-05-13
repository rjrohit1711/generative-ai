import torch
import os
import json
import sys
from PIL import Image
from rembg import remove
import numpy as np

from diffusers import DiffusionPipeline  # auto-detects correct pipeline

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils.constants as Constants

class AssetCreatorAgent:
    def __init__(self, configPath: str):
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pipeline (auto-detects any custom pipeline classes)
        self.pipe = DiffusionPipeline.from_pretrained(
            "stablediffusionapi/pixel-art-diffusion-xl",
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        ).to(self.device)
        
        # Config path
        self.configPath = configPath

    def generate_assets(self):
        print("🎨 Generating visual assets...")

        # Load game configuration
        with open(self.configPath, "r", encoding="utf-8") as f:
            config = json.load(f)
        screen_cfg = config.get("screen", {})
        screen_width  = screen_cfg.get("width")   # default fallback
        screen_height = screen_cfg.get("height")
        screen_size = [screen_width, screen_height]
        tasks = []
        for asset_info in config.get("assets_required", []):
            name = asset_info.get("asset", "unnamed")
            desc = asset_info.get("description", "")
            prompt = f"Generate high quality pixel-art assets for given description: {desc}"
            filename = f"{name.replace(' ', '_').lower()}.png"
            tasks.append((prompt, filename))

        output_dir = os.path.join("bin", "source", "assets", "sprites")
        os.makedirs(output_dir, exist_ok=True)

        for prompt, filename in tasks:
            print(f"🖼️  Generating: {prompt}")
            # Build generation kwargs, avoid passing None
            gen_kwargs = {
                "prompt": prompt,
                "num_inference_steps": 50,
                "guidance_scale": 10
            }

            result = self.pipe(**gen_kwargs)
            image = result.images[0]

            # Save and resize
            save_path = os.path.join(output_dir, filename)
            image.save(save_path)
            with Image.open(save_path) as img:
                if(filename == "scene_background.png"):
                    img = img.resize(screen_size, Image.Resampling.LANCZOS)
                else:
                    img = remove(img)
                    img.save(save_path)
                    img = strict_trim_and_zoom(save_path, 1.5)
                    img = img.resize([70,75], Image.Resampling.LANCZOS)
                img.save(save_path)

        print(f"✅ {len(tasks)} assets saved to {output_dir}")

def strict_trim_and_zoom(image_path, scale=1.0):
    image = Image.open(image_path).convert("RGBA")
    alpha = image.split()[-1]  # Get alpha channel

    # Convert alpha to numpy array
    alpha_data = np.array(alpha)

    # Find rows and columns with any visible content (alpha > threshold)
    threshold = 15  # you can adjust this
    rows = np.where(np.max(alpha_data, axis=1) > threshold)[0]
    cols = np.where(np.max(alpha_data, axis=0) > threshold)[0]

    if rows.size and cols.size:
        top, bottom = rows[0], rows[-1]
        left, right = cols[0], cols[-1]

        cropped = image.crop((left, top, right + 1, bottom + 1))

        if scale != 1.0:
            new_size = (
                int(cropped.width * scale),
                int(cropped.height * scale)
            )
            cropped = cropped.resize(new_size, Image.Resampling.LANCZOS)

        return cropped

if __name__ == "__main__":
    agent = AssetCreatorAgent(Constants.GAME_CONFIGV2)
    agent.generate_assets()
