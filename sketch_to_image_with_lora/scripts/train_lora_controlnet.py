import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from accelerate import Accelerator
from transformers import get_scheduler
from peft import LoraConfig, get_peft_model
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    DDPMScheduler,
    AutoencoderKL,
)
from transformers import CLIPTokenizer
from sketchy_dataset import SketchyDataset

def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA Fine-tuning for ControlNet Sketch→Image"
    )
    parser.add_argument("--train_sketch_dir", type=str, required=True)
    parser.add_argument("--train_photo_dir", type=str, required=True)
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--controlnet_model_name_or_path", type=str,
                        default="lllyasviel/control_v11p_sd15_scribble")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str,
                        choices=["no", "fp16", "bf16"], default="fp16")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    accelerator = Accelerator(
        mixed_precision=(None if args.mixed_precision=="no" else args.mixed_precision)
    )
    device = accelerator.device

    # ─── 1️⃣ Load models ───────────────────────────────────────────
    # ControlNet for sketches
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_model_name_or_path, torch_dtype=torch.float16
    )

    # Stable Diffusion pipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to(device)
    
    # Freeze VAE and text_encoder
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    ).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # Scheduler & tokenizer & text encoder
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = pipe.text_encoder
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False

    # Put ControlNet & UNet in train mode
    pipe.controlnet.train()
    pipe.unet.train()

    # ─── 2️⃣ Inject LoRA adapters into UNet ─────────────────────
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_v"],
        lora_dropout=args.lora_dropout,
    )
    pipe.unet = get_peft_model(pipe.unet, lora_config)

    # ─── 3️⃣ Prepare dataset & dataloader ───────────────────────
    dataset = SketchyDataset(
        sketch_root=args.train_sketch_dir,
        photo_root=args.train_photo_dir,
        image_size=args.image_size
    )
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    # ─── 4️⃣ Optimizer ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        pipe.unet.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01)
    
    # After you create optimizer:
    total_steps = args.max_train_steps
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    # ─── 5️⃣ Prepare with Accelerator ────────────────────────────
    pipe, optimizer, dataloader = accelerator.prepare(pipe, optimizer, dataloader)

    # ─── 6️⃣ Training loop ───────────────────────────────────────
    global_step = 0
    print("Starting training loop...")

    while global_step < args.max_train_steps:
        for batch in dataloader:
            # 6.1 Move data
            sketches = batch["sketch"].to(device).to(pipe.unet.dtype)
            photos   = batch["photo"].to(device)

            # 6.2 Encode to latents
            with torch.no_grad():
                latents = vae.encode(photos).latent_dist.sample() * vae.config.scaling_factor
            latents = latents.to(pipe.unet.dtype)

            # 6.3 Add noise
            noise = torch.randn_like(latents, device=latents.device, dtype=latents.dtype)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            )
            # Create the “noisy” version of latents
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            sketches = sketches.to(pipe.unet.dtype)

            # 6. Tokenize & encode text (float32) then cast to model dtype
            input_ids = tokenizer(
                ["a photo"]*bsz, padding="max_length", truncation=True,
                max_length=tokenizer.model_max_length, return_tensors="pt"
            ).input_ids.to(device)
            encoder_states = text_encoder(input_ids)[0].to(pipe.unet.dtype)

           # — Run ControlNet
            controlnet_out = pipe.controlnet(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_states,
                controlnet_cond=sketches
            )
            down_residuals = controlnet_out.down_block_res_samples
            mid_residual = controlnet_out.mid_block_res_sample

            # — Run UNet with ControlNet guidance
            unet_out = pipe.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_states,
                down_block_additional_residuals=down_residuals,
                mid_block_additional_residual=mid_residual
            )
            noise_pred = unet_out.sample

            # 6.6 Loss & backprop
            loss = F.mse_loss(noise_pred, noise)
            accelerator.backward(loss)

            # Only step every N accumulation steps
           
            clip_grad_norm_(pipe.unet.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            if global_step % args.logging_steps == 0:
                print(f"[Step {global_step:4d}/{args.max_train_steps:4d}] loss: {loss.item():.4f}")

            if global_step % args.save_steps == 0 or global_step == args.max_train_steps:
                ckpt_dir = os.path.join(args.output_dir, f"lora_step_{global_step}")
                pipe.unet.save_pretrained(ckpt_dir, safe_serialization=True)
                print(f"Saved adapter at step {global_step} → {ckpt_dir}")

            if global_step >= args.max_train_steps:
                break

    print("✅ Training complete.")

if __name__ == "__main__":
    main()
