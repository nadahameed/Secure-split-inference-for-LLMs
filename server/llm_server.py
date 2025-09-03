import socket
import struct
import numpy as np
import os
import glob

# --- Helper functions to send/receive NumPy arrays over sockets ---
# This logic is adapted from the lecture materials [cite: 672-674, 1060-1065]
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
    # Receive header
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
    
    # Receive data
    num_bytes = np.prod(shape) * dtype.itemsize
    data = b''
    while len(data) < num_bytes:
        packet = conn.recv(num_bytes - len(data))
        if not packet: raise ConnectionError("Socket closed")
        data += packet
        
    return np.frombuffer(data, dtype=dtype).reshape(shape)

def load_weights(weights_dir):
    """Loads all .npz weight files from a directory into a dictionary."""
    print(f"Loading weights from {weights_dir}...")
    weights = {}
    for f_path in glob.glob(os.path.join(weights_dir, "*.npz")):
        layer_name = os.path.basename(f_path).replace('.npz', '')
        with np.load(f_path) as data:
            weights[layer_name] = {'weight': data['weight'], 'bias': data['bias']}
    print(f"Loaded {len(weights)} weight files.")
    return weights

def main():
    HOST = '0.0.0.0'  # Listen on all available network interfaces
    PORT = 5001
    WEIGHTS_DIR = "distilgpt2_weights"

    # 1. Load all model weights into memory
    weights = load_weights(WEIGHTS_DIR)

    # 2. Set up the socket server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {HOST}:{PORT}")
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            
            # 3. Loop to handle inference requests
            while True:
                try:
                    # Receive layer name
                    name_len_bytes = conn.recv(4)
                    if not name_len_bytes: break
                    name_len = struct.unpack('!I', name_len_bytes)[0]
                    name_bytes = b''
                    while len(name_bytes) < name_len:
                        packet = conn.recv(name_len - len(name_bytes))
                        if not packet: raise ConnectionError("Socket closed during name recv")
                        name_bytes += packet
                    layer_name = name_bytes.decode('utf-8')
                    
                    # Receive encrypted input tensor
                    encrypted_input = recv_array(conn)
                    print(f"  - Received request for layer: {layer_name} | shape: {encrypted_input.shape}")

                    # 4. Perform the linear computation
                    W = weights[layer_name]['weight']
                    b = weights[layer_name]['bias']
                    
                    # Note: PyTorch weights (from Hugging Face) often need to be transposed
                    # for standard (input @ weight) matrix multiplication.
                    if layer_name == "lm_head":
                    # LM head: [batch, seq_len, hidden] @ [vocab_size, hidden].T -> [batch, seq_len, vocab_size]
                        output = encrypted_input @ W.T + b
                    else:
                        output = encrypted_input @ W + b
                    
                    # 5. Send the result back
                    send_array(conn, output)
                    
                except ConnectionError:
                    print("Client disconnected.")
                    break
    print("Server shutting down.")

if __name__ == "__main__":
    main()