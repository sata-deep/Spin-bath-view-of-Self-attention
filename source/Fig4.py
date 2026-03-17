import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import GPT2Tokenizer, GPT2Model
from functools import partial

# --- 1. SETUP: Model, Tokenizer, and Target Head ---

MODEL_NAME = 'gpt2'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2Model.from_pretrained(MODEL_NAME).to(device)
model.eval()

# Our empirically identified "antagonistic" head
ANTI_HERO_LAYER = 3
ANTI_HERO_HEAD = 5

# A control head (e.g., an early layer, low-predictive head) for comparison
CONTROL_LAYER = 0
CONTROL_HEAD = 0

d_model = model.config.n_embd
n_heads = model.config.n_head
d_head = d_model // n_heads

# --- 2. THE CORRECT HOOK FUNCTION for Ablation ---

def pre_c_proj_ablation_hook(module, input, head_to_ablate):
    """
    This is a PRE-hook attached to the c_proj layer of a GPT2Attention block.
    It modifies the INPUT to the layer, which is the concatenated output of all heads.
    Its signature is hook(module, input).
    """
    # `input` is a tuple, where input[0] is the tensor we want to modify.
    # Shape of input[0]: (batch_size, seq_len, d_model)
    concatenated_heads_output = input[0]
    
    # Reshape to isolate the heads for easy slicing
    # (batch, seq, n_heads, d_head)
    head_outputs = concatenated_heads_output.view(
        concatenated_heads_output.shape[0], 
        concatenated_heads_output.shape[1], 
        n_heads, 
        d_head
    )
    
    # Zero out the slice corresponding to the target head.
    # This is an in-place modification.
    head_outputs[:, :, head_to_ablate, :] = 0
    
    # Pre-hooks don't need to return anything if they modify the input in-place.
    # The modified `input` tuple will be passed to the original `c_proj` layer.
    
# --- 3. THE ABLATION EXPERIMENT ---

def get_logit_diff(prompt_text, good_token, bad_token, layer_to_ablate=None, head_to_ablate=None):
    """Helper function to run one forward pass and get the logit difference."""
    
    handle = None
    if layer_to_ablate is not None and head_to_ablate is not None:
        # Create a partial function to pass the `head_to_ablate` index to the hook
        hook_fn = partial(pre_c_proj_ablation_hook, head_to_ablate=head_to_ablate)
        
        # Get the target module to attach the hook to
        target_module = model.h[layer_to_ablate].attn.c_proj
        
        # Register the hook as a FORWARD_PRE_HOOK
        handle = target_module.register_forward_pre_hook(hook_fn)
    
    with torch.no_grad():
        # Tokenize inputs for this run
        prompt_tokens = tokenizer.encode(prompt_text, return_tensors='pt').to(device)
        good_token_id = tokenizer.encode(good_token, add_prefix_space=True)[0]
        bad_token_id = tokenizer.encode(bad_token, add_prefix_space=True)[0]
        
        # Run the model (the hook will fire automatically)
        outputs = model(prompt_tokens)
        
        # Calculate final logits
        final_logits = outputs.last_hidden_state @ model.wte.weight.T
        logit_good = final_logits[0, -1, good_token_id].item()
        logit_bad = final_logits[0, -1, bad_token_id].item()
    
    if handle:
        handle.remove() # Crucial: always remove the hook after use!
        
    return logit_good - logit_bad

# --- Main experiment ---

# Use the same representative prompt as before
test_prompt = "Lions are carnivores. Cows are"
test_good_token = "herbivores"
test_bad_token = "omnivores"

# 1. Get baseline (normal) performance
baseline_diff = get_logit_diff(test_prompt, test_good_token, test_bad_token)

# 2. Get performance with Anti-Hero head ablated
anti_hero_ablated_diff = get_logit_diff(
    test_prompt, test_good_token, test_bad_token, 
    layer_to_ablate=ANTI_HERO_LAYER, 
    head_to_ablate=ANTI_HERO_HEAD
)

# 3. Get performance with Control head ablated
control_ablated_diff = get_logit_diff(
    test_prompt, test_good_token, test_bad_token, 
    layer_to_ablate=CONTROL_LAYER, 
    head_to_ablate=CONTROL_HEAD
)

print("\n--- Causal Ablation Results ---")
print(f"Prompt: '{test_prompt}...'")
print(f"Baseline Logit Difference: {baseline_diff:.4f}")
print(f"Ablating Antagonistic L{ANTI_HERO_LAYER}H{ANTI_HERO_HEAD}: {anti_hero_ablated_diff:.4f}")
print(f"Ablating Control L{CONTROL_LAYER}H{CONTROL_HEAD}:      {control_ablated_diff:.4f}")

# Our hypothesis check
if anti_hero_ablated_diff > baseline_diff:
    print("\nSUCCESS: Ablating the antagonistic head IMPROVED performance, confirming its causal role.")
else:
    print("\nNOTE: Ablating the antagonistic head did not improve performance as expected. This is also an interesting result.")

# --- 4. PLOTTING THE BAR CHART ---
# This section is unchanged and should work correctly.

labels = ['Normal Run', f'Ablate L{ANTI_HERO_LAYER}H{ANTI_HERO_HEAD}\n(Antagonistic)', f'Ablate L{CONTROL_LAYER}H{CONTROL_HEAD}\n(Control)']
values = [baseline_diff, anti_hero_ablated_diff, control_ablated_diff]

plt.figure(figsize=(8, 6))
sns.set_theme(style="whitegrid")
bar_plot = sns.barplot(x=labels, y=values, palette=['gray', 'red', 'blue'], alpha=0.8)

for i, v in enumerate(values):
    plt.text(i, v + np.sign(v)*0.1, f"{v:.2f}", ha='center', va='bottom' if v > 0 else 'top', fontweight='bold')

plt.axhline(0, color='black', linewidth=0.8)
plt.ylabel('Logit Difference ($\Delta L_{actual}$)', fontsize=12)
plt.title(f'Causal Effect of Head Ablation on Factual Recall', fontsize=14, weight='bold')
plt.suptitle(f"Prompt: '{test_prompt}...' -> '{test_good_token}'", fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.96])

output_filename = 'Fig4.png'
plt.savefig(output_filename, dpi=300)
print(f"\nBar chart saved as '{output_filename}'")
plt.show()
