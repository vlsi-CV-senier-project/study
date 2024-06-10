import torch
import torch.nn.functional as F
import time

# Ensure using GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dot_product_attention(Q, K, V, mask=None):
    scores = torch.matmul(Q, K.transpose(-2, -1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output

def measure_performance(func, args, kwargs={}, num_runs=10):
    torch.cuda.reset_peak_memory_stats(device)
    start = time.time()
    for _ in range(num_runs):
        func(*args, **kwargs)
    end = time.time()
    avg_time = (end - start) / num_runs
    memory_usage = torch.cuda.max_memory_allocated(device)  # Get peak memory usage
    torch.cuda.reset_peak_memory_stats(device)  # Reset memory usage counter
    return avg_time, memory_usage

# Tensor creation
batch_size = 2
num_heads = 4
seq_len = 16
head_dim = 64
d_model = num_heads * head_dim

Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
K = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
V = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
mask = torch.ones(batch_size, num_heads, seq_len, seq_len, device=device).bool()

# Performance measurement
custom_time, custom_memory = measure_performance(dot_product_attention, (Q, K, V, mask), {})
scaled_time, scaled_memory = measure_performance(F.scaled_dot_product_attention, (Q, K, V), {'attn_mask': mask})

print(f'Custom Dot Product Attention Time: {custom_time:.6f} seconds, Memory: {custom_memory / (1024 ** 2):.2f} MB')
print(f'PyTorch Scaled Dot Product Attention Time: {scaled_time:.6f} seconds, Memory: {scaled_memory / (1024 ** 2):.2f} MB')
