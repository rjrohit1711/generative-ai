import json
import os
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import torch

from transformers import (
    GPTNeoForCausalLM,
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType

# 1) Load your JSON array
with open(r"dataset/game_design_data.json", "r", encoding="utf-8") as f:
    all_games = json.load(f)

# 2) Split into train/validation
train_list, val_list = train_test_split(all_games, test_size=0.2, random_state=42)
ds = DatasetDict({
    "train": Dataset.from_list(train_list),
    "validation": Dataset.from_list(val_list)
})

# 3) Initialize tokenizer (and ensure pad_token exists)
MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4) Convert each JSON object into a pretty-printed JSON string
def to_json_str(example):
    example = dict(example)
    example["text"] = json.dumps(example, indent=2, ensure_ascii=False)
    return example


ds = ds.map(to_json_str, remove_columns=ds["train"].column_names)

# 5) Tokenize and set labels=input_ids for causal LM
def tokenize_fn(examples):
    tok = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=1024
    )
    tok["labels"] = tok["input_ids"].copy()  # causal LM: predict next token
    return tok

tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

for i in range(3):  # first 3 samples
    input_ids = tokenized["train"][i]["input_ids"]
    print(f"\n=== Sample {i} ===")
    print(tokenizer.decode(input_ids, skip_special_tokens=True))

# 6) Apply LoRA to GPT-Neo
device = "cuda" if torch.cuda.is_available() else "cpu"
base_model = GPTNeoForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(base_model, peft_config)
model.train()

# 7) Data collator for causal LM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 8) Training arguments
OUTPUT_DIR = "lora_gamedesigner"
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1000,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-4,
    warmup_steps=50,
    fp16=True,
    logging_steps=1,
    save_steps=40,
    save_total_limit=30,
    eval_strategy="steps",
    eval_steps=20,
    load_best_model_at_end=True
)

# 9) Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

# 10) Train & save only LoRA adapters + tokenizer
trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"LoRA adapters and tokenizer saved to {OUTPUT_DIR}/")
