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

        game_screen_cfg = config.get("screens", {}) \
                        .get("game_screen", {}) \
                        .get("screen", {})
        screen_width  = game_screen_cfg.get("width")
        screen_height = game_screen_cfg.get("height")
        screen_size = [screen_width, screen_height]

        tasks = []

        # Start Screen background
        start_bg = config.get("game_info", {}) \
                        .get("start_screen", {}) \
                        .get("background", {})
        if start_bg:
            tasks.append((start_bg.get("description", ""), start_bg.get("assert_path", "")))

        # Other Screens’ backgrounds
        for screen in config.get("screens", {}).values():
            bg = screen.get("background", {})
            if bg:
                tasks.append((bg.get("description", ""), bg.get("assert_path", "")))

        # Objects’ sprites
        for obj in config.get("objects", []):
            desc = obj.get("description", obj.get("role", obj.get("id", "")))
            sprite_path = obj.get("assert_path", "")
            tasks.append((desc, sprite_path))


        output_dir = os.path.join("assets", "sprites")
        os.makedirs(output_dir, exist_ok=True)

        for prompt, asset_path in tasks:
            print(f"🖼️  Generating: {prompt}  Path: {asset_path}")
            # Build generation kwargs, avoid passing None
            gen_kwargs = {
                "prompt": prompt,
                "num_inference_steps": 50,
                "guidance_scale": 5
            }

            result = self.pipe(**gen_kwargs)
            image = result.images[0]

            # Save and resize
            save_path = asset_path
            image.save(save_path)
            with Image.open(save_path) as img:
                if("background" in save_path):
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
    agent = AssetCreatorAgent(Constants.GAME_CONFIGV3)
    agent.generate_assets()
