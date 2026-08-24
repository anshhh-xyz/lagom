import os
import torch

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

MAX_SEQ_LENGTH = 1024
DTYPE = None
LOAD_IN_4BIT = True

MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"

OUTPUT_DIR = "lagom-deep-7b-outputs"
HF_HUB_MODEL_ID = "your-username/lagom-deep-7b"

print(f"Loading Base Model: {MODEL_NAME}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",
    mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in convos
    ]
    return {"text": texts}

dataset_path = "train_chatml.jsonl"
if not os.path.exists(dataset_path):
    dataset_path = "processed_data/train_chatml.jsonl"

dataset = load_dataset("json", data_files={"train": dataset_path}, split="train")
dataset = dataset.map(formatting_prompts_func, batched=True, num_proc=4)

print(f"Dataset successfully loaded. Total training rows: {len(dataset):,}")
print("Sample Formatted Prompt:\n" + "=" * 60)
print(dataset[0]["text"][:600] + "...")
print("=" * 60)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=4,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_ratio=0.03,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=20,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir=OUTPUT_DIR,
        report_to="none",
        group_by_length=True,
    ),
)

print("\nStarting Ultra-Fast Fine-Tuning...")
trainer_stats = trainer.train()
print(f"Training Complete! Total runtime: {trainer_stats.metrics['train_runtime'] / 60:.2f} minutes.")

FastLanguageModel.for_inference(model)

sample_ai_text = (
    "In today's digital era, artificial intelligence plays a pivotal role across multifaceted domains. "
    "Furthermore, it is important to note that machine learning algorithms facilitate optimization, "
    "thereby enhancing productivity. In conclusion, embracing AI is crucial for future advancement."
)

test_messages = [
    {
        "role": "system",
        "content": (
            "You are Lagom, a specialized humanizer AI. Your task is to rewrite the provided "
            "AI-generated text so it sounds naturally human-authored, organic, and nuanced while "
            "preserving 100% of the original factual meaning and core ideas."
        ),
    },
    {
        "role": "user",
        "content": f"Rewrite this text into natural, organic human writing.\n\n[TEXT TO HUMANIZE]:\n{sample_ai_text}",
    },
]

inputs = tokenizer.apply_chat_template(
    test_messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=512,
    use_cache=True,
    temperature=0.7,
    top_p=0.9,
)

response = tokenizer.batch_decode(outputs)
print("\n" + "=" * 60)
print("TEST HUMANIZATION RESULT:")
print("=" * 60)
print(response[0].split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip())
print("=" * 60)

model.save_pretrained("lagom_deep_lora_adapters")
tokenizer.save_pretrained("lagom_deep_lora_adapters")
print("\nLoRA Adapters saved to 'lagom_deep_lora_adapters'")
