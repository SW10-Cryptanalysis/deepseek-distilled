import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Setup Model and Tokenizer
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, # Better for performance/memory
    device_map="auto"           # Automatically uses GPU if available
)

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
    # Construct the Chat Template (Zero-Shot)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Input ciphertext:\n{ciphertext}"},
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    # Note: temperature=0.0 is set via do_sample=False
    outputs = model.generate(
        inputs, 
        max_new_tokens=4096, 
        do_sample=False, 
        temperature=1.0 # Temperature is ignored when do_sample=False
    )
    
    full_response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
    
    # DeepSeek-R1 specific: Strip the reasoning block to get just the answer
    if "</think>" in full_response:
        plaintext = full_response.split("</think>")[-1].strip()
    else:
        plaintext = full_response.strip()
        
    return plaintext

# 3. Process your dataset (dataset.jsonl)
data_path = "data/dataset.jsonl"
results = []

with open(data_path, "r") as f:
    for line in f:
        item = json.loads(line)
        # Assuming your jsonl has a field named 'ciphertext'
        ciphertext = item.get("ciphertext", "")
        
        print(f"Processing: {ciphertext[:30]}...")
        result = solve_cipher(ciphertext)
        
        results.append({
            "ciphertext": ciphertext,
            "deciphered": result
        })

# 4. Save results
with open("data/results/output.json", "w") as f:
    json.dump(results, f, indent=4)

print("Done! Results saved to data/results/output.json")