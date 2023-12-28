# Mixture of Experts
This repository contains code for implementations of different methods for modeling Mixtures of Experts (currently BlockSparse MoE).

## BlockSparse
Similar to [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1), we implement a [BlockSparse MoE](./blocksparse_moe.py) that shards each expert across parallel ranks;

First, we define methods for expert parallelism, such as the Linear method.

In the `MoE` class, the forward method propagates the input forward through the model. The `expand_and_permutate_hidden_states` method expands and permutes the hidden states based on the selected experts and routing weights. The `grouped_mlp` method applies a grouped multilayer perceptron (MLP) to the expanded hidden states. The `merge_expert_outputs` method merges the outputs of the experts.

Currently, we apply top_k routing logic, but min_p routing is in the works. We use efficient CUDA and Triton kernels to speed up the processing.

### Usage
You can install the repository by running `pip install -e .`, then import the modules:

```py
from mixture_of_experts import MoE
```

Example usage:

```python
import torch
from mixture_of_experts import MoE

# Define parameters
num_experts = 10
min_p = None
top_k = 5
hidden_size = 256
intermediate_size = 512
world_size = 1

# Instantiate the MoE model
moe_model = MoE(num_experts, min_p, top_k, hidden_size, intermediate_size, world_size)

# Assume we have some input tensor `input_tensor`
input_tensor = torch.randn(128, 10, hidden_size)

# Pass the input through the model
output = moe_model(input_tensor)
```
