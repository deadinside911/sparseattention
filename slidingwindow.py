import torch

N, D = (15, 15)
window_size = 2
tolerance = 1e-5

query_matrix = torch.rand(N, D)
keys_matrix = torch.rand(N, D)
values_matrix = torch.rand(N, D)

scores = query_matrix @ keys_matrix.T

scores /= (D ** 0.5)

# Each token can attent to itself and window_size tokens ahead of it
mask = torch.triu(torch.ones(N, N))
mask = mask - torch.triu(torch.ones(N, N), diagonal=window_size + 1)

print("mask: ", mask)

masked_scores = scores.masked_fill(mask == 0, float("-inf"))

sparse_attention_weights = torch.softmax(masked_scores, dim=1)

dense_masked_scores = scores.masked_fill(mask == 0, float("-inf"))
dense_attention_weights = torch.softmax(dense_masked_scores, dim=1)

sparse_output = sparse_attention_weights @ values_matrix
dense_output = dense_attention_weights @ values_matrix

print("sparse attention weights", sparse_attention_weights)
print("sparse output", sparse_output)

print("dense attention weights", dense_attention_weights)
print("dense output", dense_output)

print(torch.allclose(dense_output, sparse_output, atol=tolerance))
