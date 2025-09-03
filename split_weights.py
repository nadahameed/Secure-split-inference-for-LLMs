import os
import numpy as np
from transformers import AutoModelForCausalLM

# --- Config ---
MODEL_NAME = "distilgpt2"
OUTPUT_DIR = "distilgpt2_weights"

print(f"Loading pre-trained model: {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Created output directory: {OUTPUT_DIR}")

# --- Helper function to save weights ---
def save_weight(file_path, weight, bias):
    np.savez(file_path, weight=weight, bias=bias)
    print(f" - {file_path} | weight shape: {weight.shape} bias shape: {bias.shape}")

# --- Process each transformer block ---
for i, block in enumerate(model.transformer.h):
    print(f"\nProcessing Block {i}...")

    # --- Attention Layers ---
    qkv_layer = block.attn.c_attn
    qkv_weights = qkv_layer.weight.detach().numpy()
    qkv_biases = qkv_layer.bias.detach().numpy()
    hidden_size = model.config.hidden_size

    q_weights, k_weights, v_weights = np.split(qkv_weights, 3, axis=1)
    q_biases, k_biases, v_biases = np.split(qkv_biases, 3)

    # Save Q/K/V **without transposing**
    save_weight(f"{OUTPUT_DIR}/block_{i}_attn_q_proj.npz", q_weights, q_biases)
    save_weight(f"{OUTPUT_DIR}/block_{i}_attn_k_proj.npz", k_weights, k_biases)
    save_weight(f"{OUTPUT_DIR}/block_{i}_attn_v_proj.npz", v_weights, v_biases)

    # Attention output projection **without transposing**
    attn_out_layer = block.attn.c_proj
    save_weight(f"{OUTPUT_DIR}/block_{i}_attn_output_proj.npz",
                attn_out_layer.weight.detach().numpy(),
                attn_out_layer.bias.detach().numpy())

    # --- Feed-Forward Network (MLP) Layers ---
    ffn_layer_1 = block.mlp.c_fc
    ffn_layer_2 = block.mlp.c_proj

    # Save FFN weights **without transposing**
    save_weight(f"{OUTPUT_DIR}/block_{i}_ffn_layer_1.npz",
                ffn_layer_1.weight.detach().numpy(),
                ffn_layer_1.bias.detach().numpy())
    save_weight(f"{OUTPUT_DIR}/block_{i}_ffn_layer_2.npz",
                ffn_layer_2.weight.detach().numpy(),
                ffn_layer_2.bias.detach().numpy())

# --- LM head ---
lm_head = model.lm_head
bias_to_save = lm_head.bias.detach().numpy() if lm_head.bias is not None else np.zeros(lm_head.weight.shape[0])
save_weight(f"{OUTPUT_DIR}/lm_head.npz",
            lm_head.weight.detach().numpy(),
            bias_to_save)

print("\nAll weights saved successfully!")
