import os
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from peft import PeftModel

class GameDesignerAgent:
    def __init__(
        self,
        base_model: str = "EleutherAI/gpt-neo-1.3B",
        lora_path: str = "lora_gamedesigner/checkpoint-1080"
    ):
        print(f"[LOG] Loading base model {base_model}...")
        # 1) Load tokenizer and ensure a proper pad token
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None or self.tokenizer.pad_token == self.tokenizer.eos_token:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # 2) Load base model
        self.base_model = GPTNeoForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        # 3) Resize embeddings if we added a pad token
        self.base_model.resize_token_embeddings(len(self.tokenizer))

        # 4) Apply LoRA if available
        if lora_path and os.path.isdir(lora_path):
            print(f"[LOG] Applying LoRA weights from {lora_path}...")
            self.model = PeftModel.from_pretrained(self.base_model, lora_path, device_map="auto")
            print(f"[INFO] Loaded adapters with config:\n{self.model.peft_config}")
        else:
            print("[LOG] No LoRA path provided or not found. Using base model only.")
            self.model = self.base_model

        self.model.eval()
        print("[LOG] GameDesignerAgentLoRA ready!\n")

    def generate_game_description(self, concept: str) -> str:
        # A minimal one-shot example & strict JSON schema
        prompt = f"""
        You are an expert game designer. 
        Design a new video game based on the following idea:

        Idea: {concept}

        Output the game design in the following JSON format:
        {{
        "game_name": "<game_name>",
        "art_style": "<art_style>",
        "setting": "<setting>",
        "characters": ["<character1>", "<character2>", "..."],
        "mechanics": ["<mechanic1>", "<mechanic2>", "..."],
        "controls": {{"<control1>": "<key>", "<control2>": "<key>"}},
        "objectives": ["<objective1>", "<objective2>", "..."],
        "assets_required": {{"<asset_type>": ["<asset1>", "<asset2>"]}},
        "sounds_required": {{"<sound_type>": ["<sound1>", "<sound2>"]}}
        }}
        """
        
        # Tokenize with attention mask
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=1024,
            return_attention_mask=True
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=400,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        raw = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Strip everything before the first '{'
        json_part = raw[raw.find("{"):]

        # Balance braces to cut off stray text
        depth = 0
        for idx, ch in enumerate(json_part):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            if depth == 0:
                return json_part[: idx + 1]
        return json_part

if __name__ == "__main__":
    agent = GameDesignerAgent(
        base_model="EleutherAI/gpt-neo-1.3B",
        lora_path="lora_gamedesigner/checkpoint-1080"
    )
    concept = "A cozy racing game in a haunted library."
    print("\n=== Generated Game Design ===\n")
    print(agent.generate_game_description(concept))
