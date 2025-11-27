import torch
from LLM.languagemodel import BigramLanguageModel
from utils import decode, encode
import json
import os

# Find your best model
best_model = 'experiments/cosine_lr_schedule_20251126_234743/model_checkpoint.pt'  # Adjust if needed

print("🎭 Generating diverse text samples...")

# Load model
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
checkpoint = torch.load(best_model, map_location=device)

model = BigramLanguageModel()
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Test prompts
prompts = [
    "ROMEO:",
    "KING HENRY:",
    "First Citizen:",
    "JULIET:",
    "To be or not to be,",
]

temperatures = [0.5, 0.8, 1.0, 1.3]

samples = {}

for prompt in prompts:
    print(f"\n{'='*80}")
    print(f"PROMPT: {prompt}")
    print(f"{'='*80}")
    samples[prompt] = {}
    
    for temp in temperatures:
        print(f"\n--- Temperature {temp} ---")
        
        # Encode prompt
        context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        
        # Generate
        with torch.no_grad():
            generated = model.generate(context, max_new_tokens=300)
        
        text = decode(generated[0].tolist())
        print(text[:400])  # Print first 400 chars
        
        samples[prompt][f"temp_{temp}"] = text

# Save samples
os.makedirs('results', exist_ok=True)
with open('results/generated_samples.json', 'w') as f:
    json.dump(samples, f, indent=2)

print("\n✅ Samples saved to results/generated_samples.json")