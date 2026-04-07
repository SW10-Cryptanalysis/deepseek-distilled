import json
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Setup Model
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    dtype=torch.bfloat16, 
    device_map="auto"
)

# 2. Load Few-Shot Examples
def load_few_shot(path, count=2):
    examples = []
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            examples.append({
                "ciphertext": data["ciphertext"],
                "plaintext": data["plaintext"]
            })
    return random.sample(examples, count)

# 3. Construct the Message List with Few-Shot examples
def solve_cipher_few_shot(target_ciphertext, few_shot_path):
    # Get 2 random examples to show the pattern
    examples = load_few_shot(few_shot_path, count=2)
    
    messages = [
        {"role": "system", "content": "You are an expert cryptanalyst. Decipher homophonic substitution ciphers. Return ONLY the raw lowercase plaintext string. No spaces, no filler."}
    ]
    
    # Add the examples to the conversation history
    for ex in examples:
        messages.append({"role": "user", "content": f"Input ciphertext:\n{ex['ciphertext']}"})
        messages.append({"role": "assistant", "content": f"<think>\nThis is a homophonic substitution. I will map frequencies and patterns to English.\n</think>\n{ex['plaintext']}"})
    
    # Add the actual target
    messages.append({"role": "user", "content": f"Input ciphertext:\n{target_ciphertext}"})
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    # Note: We keep max_new_tokens reasonable to prevent rambling
    outputs = model.generate(
        **inputs, 
        max_new_tokens=1000, 
        do_sample=False
    )
    
    full_response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
    
    if "</think>" in full_response:
        return full_response.split("</think>")[-1].strip()
    return full_response.strip()

# 4. Run the Test
few_shot_data = "data/fewshot.jsonl"
dataset_path = "data/dataset.jsonl"

# Pick a random test case from your main dataset (Length < 200 for best results)
with open(dataset_path, "r") as f:
    valid_tests = [json.loads(line) for line in f if json.loads(line).get("length", 0) <= 400]

test_item = random.choice(valid_tests)
print(f"Testing Length: {test_item['length']}")

result = solve_cipher_few_shot(test_item["ciphertext"], few_shot_data)

print("\n--- Deciphered Result ---")
print(result)