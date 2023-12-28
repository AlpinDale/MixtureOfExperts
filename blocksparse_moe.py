from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.distributed
import triton
import triton.language as tl

import blocksparse_ops as ops

def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: Optional[Dict[str, Any]],
):
    """Set attributes on a weight tensor."""
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(
            weight, key), (f"Overwriting existing tensor attribute: {key}")
        setattr(weight, key, value)

def tensor_model_parallel_all_reduce(input_):
    """All reduce across all model parallel GPUs."""
    world_size = torch.distributed.get_world_size(
        group=torch.distributed.group.WORLD)
    if world_size == 1:
        return input_
    torch.distributed.all_reduce(input_, group=torch.distributed.group.WORLD)
    return input_

class LinearBase(ABC):

    @abstractmethod
    def create_weights(self, input_size_per_partition: int,
                       output_size_per_partition: int, input_size: int,
                       output_size: int,
                       params_dtype: torch.dtype) -> Dict[str, Any]:
        """Create weights for a linear layer."""
        raise NotImplementedError

    @abstractmethod
    def apply_weights(self,
                      weights: Dict[str, torch.Tensor],
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply the weights to the input tensor."""
        raise NotImplementedError

class Replicated(torch.nn.Module):
    """Replicated linear layers."""
    def __init__(
            self,
            input_size: int,
            output_size: int,
            bias: bool = True,
            skip_add_bias_add: bool = False,
            params_dtype: Optional[torch.dtype] = None,
            linear_method: Optional[LinearBase] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.skip_add_bias_add = skip_add_bias_add
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.linear_method = linear_method
        if self.linear_method is None:
            raise ValueError("Quantization is currently not supported.")
        self.linear_weights = self.linear_method.create_weights(
            self.input_size, self.output_size, self.input_size,
            self.output_size, self.params_dtype)
        for name, weight in self.linear_weights.items():
            if isinstance(weight, torch.Tensor):
                self.register_parameter(name, Parameter(weight))
        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size,
                            device=torch.device(f'cuda:{torch.cuda.current_device()}'),
                            dtype=self.params_dtype))
            set_weight_attrs(self.bias, {"output_dim": 0})
        else:
            self.register_parameter("bias", None)


class MoE(torch.nn.Module):
    def __init__(
            self,
            num_experts: int,
            min_p: Optional[float],
            top_k: Optional[int],
            hidden_size: int,
            intermediate_size: int,
            world_size: int,
    ):
        super().__init__()
        self.num_total_experts = num_experts
        self.min_p = min_p
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size // world_size

        self.gate = Replicated(self.hidden_size,
                               self.num_total_experts,
                               bias=False,
                               linear_method=None)
        
        self.w1s = torch.nn.Parameter(
            torch.empty(self.num_total_experts, self.hidden_size,
                        self.intermediate_size))
        self.w2s = torch.nn.Parameter(
            torch.empty(self.num_total_experts, self.intermediate_size,
                        self.hidden_size))
        self.w3s = torch.nn.Parameter(
            torch.empty(self.num_total_experts, self.hidden_size,
                        self.intermediate_size))
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        router_logits, _ = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        # TODO: Figure out min_p routing.
        if self.min_p is not None:
            raise NotImplementedError("min_p routing not implemented yet")
        if self.top_k is None:
            raise ValueError("top_k must be an integer")
        routing_weights, selected_experts = torch.topk(routing_weights,
                                                      self.top_k,
                                                      dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        expanded_hidden_states, experts_range, expanded_weights, experts_indices = \
            self.expand_and_permutate_hidden_states(
                hidden_states, selected_experts, routing_weights)
        
        expanded_hidden_states = self.grouped_mlp(expanded_hidden_states,
                                                  experts_range,
                                                  self.w1s.data,
                                                  self.w2s.data,
                                                  self.w3s.data)
        
        expanded_hidden_states.mul_(expanded_weights.unsqueeze(-1))

        tensor_model_parallel_all_reduce(expanded_hidden_states)

        return self.merge_expert_outputs(expanded_hidden_states,
                                         experts_indices).view(batch_size,
                                                               sequence_length,
                                                               hidden_size)
    
    def expand_and_permutate_hidden_states(
        self,
        hidden_states: torch.Tensor,    # [batch_size, hidden_size]
        selected_experts: torch.Tensor, # [batch_size, top_k_experts]
        routing_weights: torch.Tensor,  # [batch_size, top_k_experts]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cum_experts_range = torch.zeros(self.num_total_experts + 1,
                                        dtype=torch.int32,
                                        device=hidden_states.device)
        num_rows_per_expert = torch.zeros(self.num_total_experts,
                                          dtype=torch.int32,
                                          device=hidden_states.device)
        ops.bincount(selected_experts.view(-1), num_rows_per_expert)
        torch.cumsum(num_rows_per_expert, dim=0, out=cum_experts_range[1:])
        experts_indices = torch.argsort(selected_experts.view(-1), dim=-1)
        expanded_weights = routing_weights.view(-1)[experts_indices]
        if self.top_k is None:
            raise ValueError("top_k must be an integer")
        return hidden_states[experts_indices.div_(
            self.top_k, rounding_mode="floor"
            )], cum_experts_range, expanded_weights, experts_indices
        
    def grouped_mlp(
        self,
        expanded_hidden_states: torch.Tensor,
        cum_experts_range: torch.Tensor,
        w1s: torch.Tensor,
        w2s: torch.Tensor,
        w3s: torch.Tensor,
    ) -> torch.Tensor:
        grouped_w1_out = grouped_matmul(expanded_hidden_states,
                                        cum_experts_range, w1s, "silu")
        grouped_w3_out = grouped_matmul(expanded_hidden_states,
                                        cum_experts_range, w3s)
        grouped_w1_out.mul_(grouped_w3_out)
        return grouped_matmul(grouped_w1_out, cum_experts_range, w2s)

    def merge_expert_outputs(
            self,
            expanded_hidden_states: torch.Tensor,
            experts_indices,
    ) -> torch.Tensor:
        if self.top_k is None:
            raise ValueError("top_k must be an integer")
        out = torch.zeros(expanded_hidden_states.shape[0] // self.top_k,
                          self.hidden_size,
                          device=expanded_hidden_states.device,
                          dtype=expanded_hidden_states.dtype)
        out.index_add_(0, experts_indices, expanded_hidden_states)
        return out

@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 84,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 84,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128,
        }),
    ],
    key=['group_size'],
)
@triton.jit
def grouped_matmul_kernel(
    # device tensors of matrices pointers
    fused_input_ptr,
    cum_input_group_range,
    fused_b_ptr,
    fused_output_ptr,
    group_size,
    n,
    k,
    lda,
    ldb,
    ldc,
    # number of virtual SMs
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # get the GEMM size of the current problem
        a_offset = tl.load(cum_input_group_range + g)
        gm = tl.load(cum_input_group_range + g + 1) - a_offset
        gn = n
        gk = k
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        # iterate through the tiles in the current GEMM
        while (tile_idx >= last_problem_end
               and tile_idx < last_problem_end + num_tiles):
            # pick up a tile from the current GEMM
            k = gk
            a_ptr = fused_input_ptr + a_offset * lda
            b_ptr = fused_b_ptr + g * k * n
            c_ptr = fused_output_ptr + a_offset * ldc
            # figure out the tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # regular GEMM
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N),
                                   dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])

                a = tl.load(a_ptrs,
                            mask=offs_k[None, :] < k - kk * BLOCK_SIZE_K,
                            other=0.0)
                b = tl.load(b_ptrs,
                            mask=offs_k[None, :] < k - kk * BLOCK_SIZE_K,
                            other=0.0)
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * ldb

            if ACTIVATION == "silu":
                accumulator = silu(accumulator)
            c = accumulator.to(tl.float16)

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]
            c_mask = (offs_cm[:, None] < gm) & (offs_cn[None, :] < gn)

            tl.store(c_ptrs, c, mask=c_mask)

            tile_idx += NUM_SM

        last_problem_end = last_problem_end + num_tiles
    
@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

def grouped_matmul(
    fused_input: torch.Tensor,
    cum_group_range: torch.Tensor,
    fused_group_b: torch.Tensor,
    activation: str = ""
):
    device = torch.device("cuda")
    assert cum_group_range.shape[0] == fused_group_b.shape[0] + 1
    group_size = cum_group_range.shape[0] - 1
    output = torch.zeros(
        fused_input.shape[0],
        fused_group_b.shape[2],
        device=device,
        dtype=fused_input.dtype)
    
    grid = lambda META: (META["NUM_SM"], )
    grouped_matmul_kernel[grid](
        fused_input,
        cum_group_range,
        fused_group_b,
        output,
        group_size,
        n=fused_group_b.shape[2],
        k=fused_group_b.shape[1],
        lda=fused_input.stride(0),
        ldb=fused_group_b.stride(1),
        ldc=output.stride(0),
        ACTIVATION=activation,
    )

    return output
    