import torch

N, D = (15, 15)
window_size = 2

query_matrix = torch.rand(N, D)
keys_matrix = torch.rand(N, D)
values_matrix = torch.rand(N, D)

scores = query_matrix @ keys_matrix.T

scores /= (D ** 0.5)

# Each token can attent to itself and window_size tokens ahead of it
mask = torch.triu(torch.ones(N, N))
mask = mask - torch.triu(torch.ones(N, N), diagonal=window_size + 1)

print("mask: ", mask)

scores = scores.masked_fill(mask == 0, float("-inf"))

attention_weights = torch.softmax(scores, dim=1)

output = attention_weights @ values_matrix

print("attention weights", attention_weights)
print("output", output)
