import socket
import struct
import numpy as np
import os
import glob
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time # <-- 1. IMPORT THE TIME LIBRARY

# --- Helper functions to send/receive NumPy arrays over sockets ---
def send_array(conn, arr):
    """Sends a numpy array with its shape and dtype."""
    arr = np.asarray(arr, dtype=np.float32)
    shape_bytes = struct.pack('!I', len(arr.shape)) + \
                  b''.join([struct.pack('!I', d) for d in arr.shape])
    dtype_bytes = str(arr.dtype).encode('utf-8')
    dtype_len_bytes = struct.pack('!I', len(dtype_bytes))
    data_bytes = arr.tobytes()
    conn.sendall(dtype_len_bytes + shape_bytes + dtype_bytes + data_bytes)

def recv_array(conn):
    """Receives a numpy array with its shape and dtype."""
    header_len_bytes = conn.recv(4)
    if not header_len_bytes: raise ConnectionError("Socket closed")
    header_len = struct.unpack('!I', header_len_bytes)[0]
    shape_len_bytes = conn.recv(4)
    shape_len = struct.unpack('!I', shape_len_bytes)[0]
    shape = []
    for _ in range(shape_len):
        dim_bytes = conn.recv(4)
        shape.append(struct.unpack('!I', dim_bytes)[0])
    shape = tuple(shape)
    dtype_str_bytes = conn.recv(header_len)
    dtype_str = dtype_str_bytes.decode('utf-8')
    dtype = np.dtype(dtype_str)
    num_bytes = np.prod(shape) * dtype.itemsize
    data = b''
    while len(data) < num_bytes:
        packet = conn.recv(num_bytes - len(data))
        if not packet: raise ConnectionError("Socket closed")
        data += packet
    return np.frombuffer(data, dtype=dtype).reshape(shape)

# --- Local primitives ---
def local_layer_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var  = np.var(x, axis=-1, keepdims=True)
    xhat = (x - mean) / np.sqrt(var + eps)
    return xhat * gamma + beta

def local_gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def causal_softmax(scores):
    S = scores.shape[-1]
    mask = np.triu(np.ones((S, S), dtype=bool), k=1)
    scores = scores.copy()
    scores[..., mask] = -1e9
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    return probs

def multi_head_attention(q, k, v, n_heads):
    B, S, H = q.shape
    head_dim = H // n_heads
    def split_heads(t):
        return t.reshape(B, S, n_heads, head_dim).transpose(0, 2, 1, 3)

    qh = split_heads(q)
    kh = split_heads(k)
    vh = split_heads(v)

    scores = np.matmul(qh, kh.transpose(0,1,3,2)) / np.sqrt(head_dim)
    probs = causal_softmax(scores)
    ctx = np.matmul(probs, vh)
    ctx = ctx.transpose(0, 2, 1, 3).reshape(B, S, H)
    return ctx

# --- Main Client ---
class SecureLLMClient:
    def __init__(self, host, port, weights_dir, model_name="distilgpt2"):
        self.host = host
        self.port = port
        self.weights = self._load_weights(weights_dir)

        print(f"Loading tokenizer and model params for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        cfg = model.config
        self.hidden_size = cfg.n_embd
        self.n_heads = cfg.n_head

        self.wte = model.transformer.wte.weight.detach().numpy()
        self.wpe = model.transformer.wpe.weight.detach().numpy()

        self.ln1_gamma, self.ln1_beta, self.ln2_gamma, self.ln2_beta = [], [], [], []
        for blk in model.transformer.h:
            self.ln1_gamma.append(blk.ln_1.weight.detach().numpy())
            self.ln1_beta.append(blk.ln_1.bias.detach().numpy())
            self.ln2_gamma.append(blk.ln_2.weight.detach().numpy())
            self.ln2_beta.append(blk.ln_2.bias.detach().numpy())
        
        self.lnf_gamma = model.transformer.ln_f.weight.detach().numpy()
        self.lnf_beta  = model.transformer.ln_f.bias.detach().numpy()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Connecting to server...")
        self.sock.connect((self.host, self.port))
        print("Connected.")

    def _load_weights(self, weights_dir):
        print(f"Loading offload weights from {weights_dir}...")
        weights = {}
        for f_path in glob.glob(os.path.join(weights_dir, "*.npz")):
            layer_name = os.path.basename(f_path).replace('.npz', '')
            with np.load(f_path) as data:
                weights[layer_name] = {'weight': data['weight'], 'bias': data['bias']}
        print(f"Loaded {len(weights)} weight files.")
        return weights

    def offload_layer(self, input_data, layer_name):

        # --- 1. ENCRYPTION ---
        # A random one-time key is created and added to the input data to disguise it.
        key = np.random.randn(*input_data.shape).astype(np.float32)
        encrypted_input = input_data + key

        # --- 2. OFFLOADING ---
        # The layer name and the encrypted data are sent to the server for computation.
        name_bytes = layer_name.encode('utf-8')
        self.sock.sendall(struct.pack('!I', len(name_bytes)))
        self.sock.sendall(name_bytes)
        send_array(self.sock, encrypted_input)
        encrypted_output = recv_array(self.sock)
        
        # --- 3. DECRYPTION ---
        # The effect of the key is removed from the server's response to get the true result.
        W = self.weights[layer_name]['weight']
        if layer_name == "lm_head":
            transformed_key = key @ W.T
        else:
            transformed_key = key @ W
        
        return encrypted_output - transformed_key

    def embed_with_positions(self, input_ids):
        input_ids = np.asarray(input_ids, dtype=np.int64)
        B, S = input_ids.shape
        pos_ids = np.arange(S, dtype=np.int64)[None, :]
        x = self.wte[input_ids] + self.wpe[pos_ids]
        return x.astype(np.float32)

    def forward_blocks(self, x):
        num_blocks = len(self.ln1_gamma)
        for i in range(num_blocks):
            # Attn
            x_norm1 = local_layer_norm(x, self.ln1_gamma[i], self.ln1_beta[i])
            q = self.offload_layer(x_norm1, f"block_{i}_attn_q_proj")
            k = self.offload_layer(x_norm1, f"block_{i}_attn_k_proj")
            v = self.offload_layer(x_norm1, f"block_{i}_attn_v_proj")
            attn_ctx = multi_head_attention(q, k, v, self.n_heads)
            attn_out = self.offload_layer(attn_ctx, f"block_{i}_attn_output_proj")
            x = x + attn_out

            # MLP
            x_norm2 = local_layer_norm(x, self.ln2_gamma[i], self.ln2_beta[i])
            ff1 = self.offload_layer(x_norm2, f"block_{i}_ffn_layer_1")
            ff1_gelu = local_gelu(ff1)
            ff2 = self.offload_layer(ff1_gelu, f"block_{i}_ffn_layer_2")
            x = x + ff2
        return x

    def generate(self, prompt, max_new_tokens=20):
        ids = self.tokenizer(prompt, return_tensors="np")["input_ids"][0].tolist()
        for _ in range(max_new_tokens):
            x = self.embed_with_positions(np.array([ids], dtype=np.int64))
            x = self.forward_blocks(x)
            x = local_layer_norm(x, self.lnf_gamma, self.lnf_beta)
            logits = self.offload_layer(x, "lm_head")[0, -1]
            next_id = int(np.argmax(logits))
            ids.append(next_id)
        return self.tokenizer.decode(ids)

    def close(self):
        self.sock.close()
        print("Connection closed.")

if __name__ == "__main__":
    SERVER_HOST = "YOUR_COMPUTER_IP_ADDRESS"
    SERVER_PORT = 5001
    WEIGHTS_DIR = "distilgpt2_weights"

    client = SecureLLMClient(SERVER_HOST, SERVER_PORT, WEIGHTS_DIR)
    
    # --- 2. ADD TIMERS AROUND THE GENERATION CALL ---
    start_time = time.time()
    generated_text = client.generate("Global climate change is primarily caused by", max_new_tokens=20)
    #test 1: Global climate change is primarily caused by
    #output: Global climate change is primarily caused by the global warming of the planet's atmosphere. (30.25 seconds)
    #test 2: In recent years, researchers have explored methods to make machine learning systems more efficient on small devices. These methods often involve a trade-off between accuracy and performance, which is particularly important for applications where privacy and latency are critical.
    #output (100 max new tokens): In recent years, researchers have explored methods to make machine learning systems more efficient on small devices. These methods often involve a trade-off between accuracy and performance, which is particularly important for applications where privacy and latency are critical.
    #time^ : 510 seconds
    end_time = time.time()
    
    print("\n--- GENERATION ---")
    print(generated_text)
    
    # --- 3. PRINT THE TOTAL TIME ---
    print(f"\n--- PERFORMANCE ---")
    print(f"Total Inference Time: {end_time - start_time:.2f} seconds")
    
    client.close()