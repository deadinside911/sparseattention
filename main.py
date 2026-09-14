import torch

N, D = (3, 3)

query_matrix = torch.rand(N, D)
keys_matrix = torch.rand(N, D)
values_matrix = torch.rand(N, D)

scores = query_matrix @ keys_matrix.T

scores /= (D ** 0.5)

mask = torch.ones(N, N)
scores = scores.masked_fill(mask == 0, float("-inf"))

attention_weights = torch.softmax(scores, dim=1)

output = attention_weights @ values_matrix

print("attention weights", attention_weights)
print("output", output)
