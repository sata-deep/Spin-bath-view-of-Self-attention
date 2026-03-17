import torch
import pandas as pd
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
from tqdm import tqdm # For a nice progress bar

# --- 1. SETUP: Load Model and Define Dataset ---

MODEL_NAME = 'gpt2'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2Model.from_pretrained(MODEL_NAME).to(device)
model.eval()

# Extract model dimensions for looping
n_layers = model.config.n_layer
n_heads = model.config.n_head
d_model = model.config.n_embd
d_head = d_model // n_heads

# --- Our new dataset of 20 diverse prompts ---
prompt_dataset = [
    {"prompt": "The capital of France is Paris. The capital of Germany is", "good": "Berlin", "bad": "Rome"},
    {"prompt": "The laws of physics are same in all inertial", "good": "frames", "bad": "bodies"},
    {"prompt": "The author of Hamlet was Shakespeare. The author of The Odyssey was", "good": "Homer", "bad": "Plato"},
    {"prompt": "An apple is a fruit. A carrot is a", "good": "vegetable", "bad": "mineral"},
    {"prompt": "The currency of Japan is the Yen. The currency of the United Kingdom is the", "good": "Pound", "bad": "Dollar"},
    {"prompt": "A dog says woof. A cat says", "good": "meow", "bad": "moo"},
    {"prompt": "The planet closest to the Sun is Mercury. The second closest is", "good": "Venus", "bad": "Mars"},
    {"prompt": "To drive a nail, you use a hammer. To tighten a screw, you use a", "good": "screwdriver", "bad": "wrench"},
    {"prompt": "The opposite of hot is", "good": "cold", "bad": "warm"},
    {"prompt": "The chemical symbol for water is H2O. The chemical symbol for salt is", "good": "NaCl", "bad": "KCl"},
    {"prompt": "A triangle has three sides. A square has", "good": "four", "bad": "five"},
    {"prompt": "The first letter of the alphabet is A. The last letter is", "good": "Z", "bad": "Y"},
    {"prompt": "In the Northern Hemisphere, summer is hot. Winter is", "good": "cold", "bad": "mild"},
    {"prompt": "Lions are carnivores. Cows are", "good": "herbivores", "bad": "omnivores"},
    {"prompt": "The main ingredient in bread is flour. The main ingredient in wine is", "good": "grapes", "bad": "apples"},
    {"prompt": "The color of the sky on a clear day is blue. The color of grass is", "good": "green", "bad": "yellow"},
    {"prompt": "One dozen is equal to twelve. Two dozen is equal to", "good": "twenty", "bad": "thirty"},
    {"prompt": "The Pacific is an ocean. The Amazon is a", "good": "river", "bad": "lake"},
    {"prompt": "A story has a beginning, a middle, and an", "good": "end", "bad": "epilogue"},
    {"prompt": "To see, you use your eyes. To hear, you use your", "good": "ears", "bad": "nose"},
]

# --- 2. DATA COLLECTION LOOP ---

results_list = []
print("Starting data collection across all prompts and heads...")

# Use tqdm for a progress bar over the prompts
for i, scenario in enumerate(tqdm(prompt_dataset, desc="Processing Prompts")):
    prompt_text = scenario['prompt']
    good_token_str = scenario['good']
    bad_token_str = scenario['bad']
    
    # Tokenize inputs
    prompt_tokens = tokenizer.encode(prompt_text, return_tensors='pt').to(device)
    good_token_id = tokenizer.encode(good_token_str, add_prefix_space=True)[0]
    bad_token_id = tokenizer.encode(bad_token_str, add_prefix_space=True)[0]
    # Note: add_prefix_space=True is important for many tokens that follow a space.

    with torch.no_grad():
        # --- Get ΔL_actual once per prompt ---
        full_model_outputs = model(prompt_tokens)
        final_logits = full_model_outputs.last_hidden_state @ model.wte.weight.T
        actual_logit_good = final_logits[0, -1, good_token_id].item()
        actual_logit_bad = final_logits[0, -1, bad_token_id].item()
        delta_l_actual = actual_logit_good - actual_logit_bad
        
        # --- Get ΔL_theory for each head ---
        S = model.wte.weight[prompt_tokens].squeeze(0)
        S_good_full = model.wte.weight[good_token_id]
        S_bad_full = model.wte.weight[bad_token_id]

        for layer_idx in range(n_layers):
            layer = model.h[layer_idx]
            W_qkv = layer.attn.c_attn.weight
            W_Q_all, W_K_all, W_V_all = W_qkv.split(d_model, dim=1)
            
            for head_idx in range(n_heads):
                # Extract weights for the current head
                start, end = head_idx * d_head, (head_idx + 1) * d_head
                W_Q = W_Q_all[:, start:end]
                W_K = W_K_all[:, start:end]
                W_V = W_V_all[:, start:end]
                W_O = layer.attn.c_proj.weight[:, start:end]

                # Theoretical calculation
                Q = S @ W_Q
                K = S @ W_K
                Omega = Q @ K.T / np.sqrt(d_head)
                attention_probs = torch.nn.functional.softmax(Omega[-1, :], dim=0)
                
                V = S @ W_V
                N_0_head = attention_probs @ V
                N_0_projected = N_0_head @ W_O.T

                theoretical_logit_good = N_0_projected @ S_good_full
                theoretical_logit_bad = N_0_projected @ S_bad_full
                delta_l_theory = (theoretical_logit_good - theoretical_logit_bad).item()

                # Store result
                results_list.append({
                    "prompt_id": i,
                    "prompt": prompt_text,
                    "layer": layer_idx,
                    "head": head_idx,
                    "delta_l_theory": delta_l_theory,
                    "delta_l_actual": delta_l_actual
                })

# --- 3. SAVE RESULTS TO CSV ---
results_df = pd.DataFrame(results_list)
output_filename = 'head_analysis_results.csv'
results_df.to_csv(output_filename, index=False)

print(f"\nData collection complete. Results saved to '{output_filename}'")
print(f"DataFrame shape: {results_df.shape}")
print("\nSample of the data:")
print(results_df.head())
