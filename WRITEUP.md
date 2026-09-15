# writeup

## 1.1 Implementing manual attention

Implementing "scaled dot product attention" as described in [Attention is All You Need](https://arxiv.org/pdf/1706.03762), using PyTorch.

## 1.2 Implementing sparsity patterns

### Sliding window

Representing allowed interactions with 1s and disallowed interactions with 0s, making the torch using numpy

```python
# Each token can attent to itself and window_size tokens ahead of it
mask = torch.triu(torch.ones(N, N))
mask = mask - torch.triu(torch.ones(N, N), diagonal=window_size + 1)
```

## 1.3 Correctness harness

Checking if sparse matches dense on unmasked positions within tolerance

```python
print(torch.allclose(dense_output, sparse_output, atol=tolerance))
```
