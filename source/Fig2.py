import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2Model

# --- 1. SETUP: Load Model and Define Target Head ---

MODEL_NAME = 'gpt2'
RANDOM_SEED = 42

# --- >>>> Point to the new "antagonistic" head <<<< ---
TARGET_LAYER = 3
TARGET_HEAD = 5
# --- >>>> ------------------------------------ <<<< ---

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2Model.from_pretrained(MODEL_NAME)
model.eval()

# Extract dimensions and weights for the target head
n_heads = model.config.n_head
d_model = model.config.n_embd
d_head = d_model // n_heads

print(f"--- Model Configuration ---")
print(f"Model: {MODEL_NAME}")
print(f"Generating Phase Boundary plot for: Layer {TARGET_LAYER}, Head {TARGET_HEAD}\n")

with torch.no_grad():
    W_E = model.wte.weight
    layer = model.h[TARGET_LAYER]
    W_qkv = layer.attn.c_attn.weight
    W_Q_all, W_K_all, W_V_all = W_qkv.split(d_model, dim=1)
    start, end = TARGET_HEAD * d_head, (TARGET_HEAD + 1) * d_head
    W_Q = W_Q_all[:, start:end]
    W_K = W_K_all[:, start:end]
    W_V = W_V_all[:, start:end]
    W_O = layer.attn.c_proj.weight[:, start:end]

# --- 2. SCENARIO DEFINITION ---
prompt = "The capital of France is Paris. The capital of Germany is"
good_token_str = "Berlin"
bad_token_str = "Rome"

prompt_tokens = tokenizer.encode(prompt, return_tensors='pt')
good_token_id = tokenizer.encode(good_token_str)[0]
bad_token_id = tokenizer.encode(bad_token_str)[0]

# --- 3. THEORETICAL CALCULATION ---
# --- 3. THEORETICAL CALCULATION ---
with torch.no_grad():
    S = W_E[prompt_tokens].squeeze(0)
    Q = S @ W_Q
    K = S @ W_K
    Omega = Q @ K.T / np.sqrt(d_head)
    attention_probs = torch.nn.functional.softmax(Omega[-1, :], dim=0)
    
    # Corrected Line: Use W_V which was defined above
    V = S @ W_V 
    
    N_0_head = attention_probs @ V
    S_good = W_E[good_token_id] @ W_V
    S_bad = W_E[bad_token_id] @ W_V
# --- 4. 2D PROJECTION & VISUALIZATION (IMPROVED VERSION) ---
print("--- Generating Improved Plot ---")
v_diff = S_good - S_bad
v_diff /= torch.linalg.norm(v_diff)
N_0_norm = N_0_head / torch.linalg.norm(N_0_head)
proj_N_on_v_diff = torch.dot(N_0_norm, v_diff) * v_diff
v_ortho = N_0_norm - proj_N_on_v_diff
v_ortho /= torch.linalg.norm(v_ortho)

x_good = torch.dot(S_good, v_ortho).item()
y_good = torch.dot(S_good, v_diff).item()
x_bad = torch.dot(S_bad, v_ortho).item()
y_bad = torch.dot(S_bad, v_diff).item()
x_N0 = torch.dot(N_0_head, v_ortho).item()
y_N0 = torch.dot(N_0_head, v_diff).item()

# Dynamic axis limits
all_x = [x_good, x_bad, 0]
all_y = [y_good, y_bad, 0]
center_x = np.mean(all_x)
center_y = np.mean(all_y)
span_x = (np.max(all_x) - np.min(all_x)) * 1.5
span_y = (np.max(all_y) - np.min(all_y)) * 1.5
plot_span = max(span_x, span_y, 5.0)
xlim = (center_x - plot_span / 2, center_x + plot_span / 2)
ylim = (center_y - plot_span / 2, center_y + plot_span / 2)

res = 200
x_range = np.linspace(xlim[0], xlim[1], res)
y_range = np.linspace(ylim[0], ylim[1], res)
grid_x, grid_y = np.meshgrid(x_range, y_range)
dot_products = grid_x * x_N0 + grid_y * y_N0

# --- 5. PLOTTING ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(8, 8))

max_abs_val = np.max(np.abs(dot_products))
levels = np.linspace(-max_abs_val, max_abs_val, 50)
contour = ax.contourf(grid_x, grid_y, dot_products, cmap='coolwarm', levels=levels, alpha=0.7)
fig.colorbar(contour, ax=ax, label=f'Alignment with $N_0$ (Head {TARGET_HEAD})')
ax.contour(grid_x, grid_y, dot_products, levels=[0], colors='black', linestyles='--')

ax.plot(x_good, y_good, 'o', markersize=14, color='green', label=f"'{good_token_str}' (Favorable)", markeredgecolor='black')
ax.plot(x_bad, y_bad, 's', markersize=14, color='darkorange', label=f"'{bad_token_str}' (Unfavorable)", markeredgecolor='black')
ax.arrow(0, 0, x_N0, y_N0, head_width=0.15, head_length=0.2, fc='k', ec='k', label='$N_0$ Vector')

ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel('Projection onto Orthogonal Axis')
ax.set_ylabel('Projection onto Decision Axis ($S_{good} - S_{bad}$)')
ax.set_title(f"Decision Landscape of Antagonistic Head L{TARGET_LAYER}H{TARGET_HEAD}\nPrompt: '{prompt}...'")

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=f"'{good_token_str}'", markerfacecolor='g', markersize=12),
    Line2D([0], [0], marker='s', color='w', label=f"'{bad_token_str}'", markerfacecolor='darkorange', markersize=12),
    Line2D([0], [0], color='k', lw=0, marker='$\\rightarrow$', markersize=12, label='$N_0$ Vector')
]
ax.legend(handles=legend_elements, loc='upper left')

ax.grid(True)
ax.set_aspect('equal', adjustable='box')
plt.tight_layout()

output_filename = 'Fig2.png'
plt.savefig(output_filename, dpi=300)
print(f"New Figure 2 saved as '{output_filename}'")
plt.show()

# --- 6. FINAL VALIDATION CHECK ---
with torch.no_grad():
    N_0_projected = N_0_head @ W_O.T
    S_good_full = W_E[good_token_id]
    S_bad_full = W_E[bad_token_id]
    theoretical_logit_good = N_0_projected @ S_good_full
    theoretical_logit_bad = N_0_projected @ S_bad_full
    theoretical_diff = theoretical_logit_good - theoretical_logit_bad
    print(f"\nTheoretical Logit Difference for L{TARGET_LAYER}H{TARGET_HEAD}: {theoretical_diff.item():.4f}")
