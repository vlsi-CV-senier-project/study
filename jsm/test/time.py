import torch
import torch.nn.functional as F
import time

def dot_product_attention(Q, K, V, mask=None):
    scores = torch.matmul(Q, K.transpose(-2, -1))
    d_k = Q.size(-1)
    scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output


def measure_time(func, *args, num_runs=10):
    start = time.time()
    for _ in range(num_runs):
        func(*args)
    end = time.time()
    avg_time = (end - start) / num_runs
    return avg_time


batch_size = 2
num_heads = 4
seq_len = 16
head_dim = 64

Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
K = torch.randn(batch_size, num_heads, seq_len, head_dim)
V = torch.randn(batch_size, num_heads, seq_len, head_dim)

mask = torch.ones(batch_size, num_heads, seq_len, seq_len)

custom_time = measure_time(dot_product_attention, Q, K, V, mask)
print(f'Dot Product Attention Time: {custom_time:.6f} seconds')


pytorch_time = measure_time(F.scaled_dot_product_attention, Q, K, V, mask)
print(f'PyTorch Scaled Dot Product Attention Time: {pytorch_time:.6f} seconds')