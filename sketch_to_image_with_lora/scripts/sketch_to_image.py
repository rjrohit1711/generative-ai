#!/usr/bin/env python
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from peft import PeftModel

def load_lora_pipeline(
    base_model="runwayml/stable-diffusion-v1-5",
    controlnet_model="lllyasviel/control_v11p_sd15_scribble",
    lora_adapter_path="lora_output_config/lora_config_1/"
):
    # 1️⃣ Load ControlNet and base pipeline
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model, torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to("cuda")
    
    # 2️⃣ Inject the LoRA weights into UNet
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_adapter_path)
    
    # 3️⃣ Make sure unet & controlnet are in eval mode
    pipe.unet.eval()
    pipe.controlnet.eval()
    
    return pipe

def run_inference(pipe, sketch_path, prompt, image_size=256, steps=50, scale=8.0):
    # Load & preprocess sketch
    sketch = Image.open(sketch_path).convert("RGB").resize((image_size, image_size))
    
    # Run the pipeline
    out = pipe(
        prompt=prompt,
        image=sketch,
        num_inference_steps=steps,
        guidance_scale=scale,
        controlnet_conditioning_scale=1.2
    )
    
    # Save the first generated image
    image = out.images[0]
    image.show()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sketch", type=str, required=True, help="Path to input sketch PNG")
    p.add_argument("--prompt", type=str, default="Get a high resolution photo of sketch", help="Text prompt")
    p.add_argument("--steps", type=int, default=50, help="Inference steps")
    p.add_argument("--scale", type=float, default=8.0, help="Guidance scale")
    args = p.parse_args()

    pipe = load_lora_pipeline()
    run_inference(pipe, args.sketch, args.prompt,
                  steps=args.steps, scale=args.scale)
