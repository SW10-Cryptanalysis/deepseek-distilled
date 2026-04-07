import json
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Setup Model and Tokenizer
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, # Better for performance/memory
    device_map="auto"           # Automatically uses GPU if available
)

MAX_CIPHER_LENGTH = 3000

# 2. Define your System Prompt from the config
SYSTEM_PROMPT = """# Task
You are an expert cryptanalyst. You are provided with a ciphertext generated via homophonic substitution. Decipher the text and return ONLY the raw plaintext. Do not include conversational filler, labels, or explanations.
# Encoding Rules
- The ciphertext is a homophonic substitution cipher.
- **Homophonic** means each plaintext letter can be represented by more than one number.
- Each number in the ciphertext, separated by a single space, represents exactly one plaintext letter.
- The plaintext contains only lowercase English letters (a-z) with no spaces or punctuation.
- The ciphertext and plaintext are always the same length.
- The plaintext is English text.

# Decoding Strategy
1. Analyse the frequency distribution of numbers in the ciphertext.
2. Group numbers that likely represent the same letter.
3. Map symbol frequencies to expected English letter frequencies.
4. Iteratively refine until a coherent English string emerges.

# Output Format
Your entire response must be only the plaintext string of lowercase letters (a-z).
-No spaces or punctuation.
-No preamble, labels, or trailing commentary.
"""

def solve_cipher(ciphertext):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Input ciphertext:\n{ciphertext}"},
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs, 
        max_new_tokens=4096, 
        do_sample=False
    )
    
    generated_ids = outputs[0][inputs['input_ids'].shape[-1]:]
    full_response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Robust splitting logic
    if "<think>" in full_response and "</think>" in full_response:
        # Standard R1 output: [Reasoning] </think> [Answer]
        parts = full_response.split("</think>")
        reasoning = parts[0].replace("<think>", "").strip()
        plaintext = parts[1].strip()
    elif "</think>" in full_response:
        # Sometimes the opening tag gets swallowed
        parts = full_response.split("</think>")
        reasoning = parts[0].strip()
        plaintext = parts[1].strip()
    else:
        # No reasoning block found at all
        reasoning = "No separate reasoning block generated."
        plaintext = full_response.strip()
        
    return reasoning, plaintext

# 3. Process your dataset (dataset.jsonl)
data_path = "data/dataset.jsonl"
with open(data_path, "r") as f:
    # Filter for items that fit your length criteria first
    valid_entries = [json.loads(line) for line in f if json.loads(line).get("length", 0) <= MAX_CIPHER_LENGTH]

if not valid_entries:
    print("No entries found matching the length criteria!")
else:
    # 3. Pick ONE random entry
    random_item = random.choice(valid_entries)
    ciphertext = random_item.get("ciphertext", "")
    actual_length = random_item.get("length")
    
    print(f"--- Running Random Test (Length: {actual_length}) ---")
    
    reasoning, plaintext = solve_cipher(ciphertext)
    
    print("\n--- Model Reasoning ---")
    print(reasoning)
    print("\n--- Final Output ---")
    print(plaintext)