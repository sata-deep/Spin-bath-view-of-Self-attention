import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 1.5
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# --- Configuration ---
PROMPTS = [
    "The capital of France is Paris. The capital of Germany is",
    "The laws of physics are same in all inertial",
    "The author of Hamlet was Shakespeare. The author of The Odyssey was",
    "An apple is a fruit. A carrot is a",
    "The currency of Japan is the Yen. The currency of the United Kingdom is the",
    "A dog says woof. A cat says",
    "The planet closest to the Sun is Mercury. The second closest is",
    "To drive a nail you use a hammer. To tighten a screw you use a",
    "The opposite of hot is",
    "The chemical symbol for water is H2O. The chemical symbol for salt is",
    "A triangle has three sides. A square has",
    "The first letter of the alphabet is A. The last letter is",
    "In the Northern Hemisphere, summer is hot. Winter is",
    "Lions are carnivores. Cows are",
    "The main ingredient in bread is flour. The main ingredient in wine is",
    "The color of the sky on a clear day is blue. The color of grass is",
    "One dozen is equal to twelve. Two dozen is equal to",
    "The Pacific is an ocean. The Amazon is a",
    "A story has a beginning, a middle, and an",
    "To see you use your eyes. To hear you use your"
]

MODEL_NAME = 'gpt2' # You can try 'gpt2-medium', 'gpt2-large', etc.
TEMPERATURE_VALUES = np.linspace(0.1, 2.5, 30) # Range of temperatures to test
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPSILON = 1e-9 # For numerical stability in log

# --- Load Model and Tokenizer ---
print(f"Loading model: {MODEL_NAME}...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval() # Set to evaluation mode (disables dropout, etc.)

# Add padding token if it's not already defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
print(f"Model loaded on {DEVICE}.")

# --- Helper Function to Calculate Order Parameters ---
def get_order_parameters(logits, temperature):
    """
    Calculates entropy and P(top-1) for given logits and temperature.
    Args:
        logits (torch.Tensor): Raw logits from the model for the next token.
        temperature (float): The temperature to apply.
    Returns:
        tuple: (entropy, p_top1)
    """
    if temperature == 0: # Avoid division by zero, treat as very low temp
        temperature = EPSILON
        
    scaled_logits = logits / temperature
    probabilities = torch.softmax(scaled_logits, dim=-1)
    
    # Entropy: S = - Σ p_i log2(p_i)
    # Add EPSILON to probabilities before log to avoid log(0)
    log_probs = torch.log2(probabilities + EPSILON)
    entropy = -torch.sum(probabilities * log_probs).item()
    
    # P(top-1)
    p_top1 = torch.max(probabilities).item()
    
    return entropy, p_top1

# --- Main Loop for Sweeping Temperatures ---
avg_entropies = []
avg_p_top1s = []

print("\nStarting temperature sweep...")
for temp_idx, temp in enumerate(TEMPERATURE_VALUES):
    current_temp_entropies = []
    current_temp_p_top1s = []
    
    for prompt_idx, prompt_text in enumerate(PROMPTS):
        inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Get logits for the *next* token (after the last token in the prompt)
            next_token_logits = outputs.logits[0, -1, :] 
            
        entropy, p_top1 = get_order_parameters(next_token_logits, temp)
        current_temp_entropies.append(entropy)
        current_temp_p_top1s.append(p_top1)
        
    avg_entropy_for_temp = np.mean(current_temp_entropies)
    avg_p_top1_for_temp = np.mean(current_temp_p_top1s)
    
    avg_entropies.append(avg_entropy_for_temp)
    avg_p_top1s.append(avg_p_top1_for_temp)
    
    print(f"Temp {temp:.2f} ({temp_idx+1}/{len(TEMPERATURE_VALUES)}): Avg Entropy = {avg_entropy_for_temp:.3f} bits, Avg P(top-1) = {avg_p_top1_for_temp:.3f}")

print("Temperature sweep complete.")

# --- Plotting Results ---
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Global Generation Temperature',fontsize=14)
ax1.set_ylabel('Avg. Next-Token Entropy (bits)', color=color,fontsize=14)
ax1.plot(TEMPERATURE_VALUES, avg_entropies, color=color, marker='o', label='Avg. Entropy')
ax1.tick_params(axis='y', labelcolor=color)
#ax1.grid(True, linestyle='--', alpha=0.7)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Avg. P(top-1)', color=color,fontsize=14)  # we already handled the x-label with ax1
ax2.plot(TEMPERATURE_VALUES, avg_p_top1s, color=color, marker='o', linestyle='--', label='Avg. P(top-1)')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
#plt.title(f'Order Parameters vs. Generation Temperature for {MODEL_NAME}')
# Adding a single legend for both lines (optional, can also use ax1.legend() and ax2.legend() separately)
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='center right')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig("Fig3.png",dpi=300)

plt.show()

print("\n--- Interpretation for Physics Background ---")
print("1. Avg. Next-Token Entropy (Red Line):")
print("   - This is analogous to thermodynamic entropy. As temperature (T) increases,")
print("     the model's next-token predictions become more uncertain and spread out,")
print("     leading to higher entropy. This is like a physical system exploring more")
print("     microstates at higher temperatures.")
print("   - At very low T, entropy is low, meaning the model is very 'certain' about")
print("     its top prediction (like a system in its ground state or a highly ordered phase).")
print("\n2. Avg. P(top-1) (Blue Line):")
print("   - This represents the probability of the single most likely next token. It's")
print("     inversely related to entropy and can be thought of as a measure of 'order'")
print("     or 'magnetization' in one specific direction (the most probable outcome).")
print("   - As T increases, P(top-1) decreases, indicating the model is less committed")
print("     to any single prediction (disordering effect of temperature).")
print("   - At low T, P(top-1) is high, indicating a strong preference for one outcome.")
print("\nLook for:")
print(" - Smooth transitions: Indicating a gradual change in the model's output character.")
print(" - 'Knees' or inflection points: These might suggest regions where the temperature")
print("   has a more pronounced effect, perhaps analogous to crossing a diffuse phase boundary.")
print(" - Relationship between the curves: They should generally show opposite trends.")
