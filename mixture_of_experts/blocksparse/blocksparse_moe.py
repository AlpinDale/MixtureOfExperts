from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

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


class FullLinearMethod(LinearBase):

    def __init__(self, separate_bias_add: bool = False):
        self.separate_bias_add = separate_bias_add

    def create_weights(self, input_size_per_partition: int,
                       output_size_per_partition: int, input_size: int,
                       output_size: int,
                       params_dtype: torch.dtype) -> Dict[str, Any]:
        weight = Parameter(torch.empty(output_size_per_partition,
                                       input_size_per_partition,
                                       device=torch.cuda.current_device(),
                                       dtype=params_dtype),
                           requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        return {"weight": weight}

    def apply_weights(self,
                      weights: Dict[str, torch.Tensor],
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        weight = weights["weight"]
        x = x.to(weight.device)
        if self.separate_bias_add:
            if bias:
                return F.linear(x, weight) + bias
            return F.linear(x, weight)
        return F.linear(x, weight, bias)

class Replicated(torch.nn.Module):
    """Replicated linear layers."""
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        linear_method: Optional[LinearBase] = None,
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        if linear_method is None:
            linear_method = FullLinearMethod()
        self.linear_method = linear_method
        self.linear_weights = self.linear_method.create_weights(
            self.input_size, self.output_size, self.input_size,
            self.output_size, self.params_dtype)
        for name, weight in self.linear_weights.items():
            if isinstance(weight, torch.Tensor):
                self.register_parameter(name, weight)
        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size,
                            device=torch.cuda.current_device(),
                            dtype=self.params_dtype))
            set_weight_attrs(self.bias, {"output_dim": 0})
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias if not self.skip_bias_add else None
        output = self.linear_method.apply_weights(self.linear_weights, x, bias)
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


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
        self.world_size = world_size
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
        # router_logits: (batch * sequence_length, n_experts)
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
                                         experts_indices).view(batch_size, sequence_length, hidden_size)

    def expand_and_permutate_hidden_states(
        self,
        hidden_states: torch.Tensor,  # [batch_size, hidden_size]
        selected_experts: torch.Tensor,  # [batch_size, top_k_experts]
        routing_weights: torch.Tensor,  # [batch_size, top_k_experts]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cum_experts_range = torch.zeros(self.num_total_experts + 1,
                                        dtype=torch.int32,
                                        device=hidden_states.device)
        num_rows_per_expert = torch.zeros(self.num_total_experts, dtype=torch.int32,
                                        device=hidden_states.device)
        ops.moe_bincount(selected_experts.view(-1), num_rows_per_expert)
        torch.cumsum(num_rows_per_expert, dim=0, out=cum_experts_range[1:])
        experts_indices = torch.argsort(selected_experts.view(-1), dim=-1)
        assert experts_indices.max() < routing_weights.numel(), "Index out of bounds"
        expanded_weights = routing_weights.view(-1)[experts_indices]
        return hidden_states[experts_indices.div_(
            self.top_k, rounding_mode="floor"
        )], cum_experts_range, expanded_weights, experts_indices

    def grouped_mlp(
        self,
        expanded_hidden_states: torch.
        Tensor,  # [batch_size * top_k_experts, hidden_size]
        cum_experts_range: torch.Tensor,  # [num_experts + 1]
        w1s: torch.Tensor,  # [num_experts, hidden_size, ffn_dim]
        w2s: torch.Tensor,  # [num_experts, ffn_dim, hidden_size]
        w3s: torch.Tensor,  # [num_experts, hidden_size, ffn_dim]
    ) -> torch.Tensor:  # [batch_size * top_k_experts, hidden_size]
        grouped_w1_out = grouped_matmul(expanded_hidden_states,
                                        cum_experts_range, w1s, "silu")
        grouped_w3_out = grouped_matmul(expanded_hidden_states,
                                        cum_experts_range, w3s)
        grouped_w1_out.mul_(grouped_w3_out)
        return grouped_matmul(grouped_w1_out, cum_experts_range, w2s)

    def merge_expert_outputs(
            self,
            expanded_hidden_states: torch.
        Tensor,  # [batch_size * top_k_experts, hidden_size]
            expert_indicies,  # [batch_size * top_k_experts]
    ) -> torch.Tensor:
        out = torch.zeros(expanded_hidden_states.shape[0] // self.top_k,
                          self.hidden_size,
                          device=expanded_hidden_states.device,
                          dtype=expanded_hidden_states.dtype)
        out.index_add_(0, expert_indicies, expanded_hidden_states)
        return out


@triton.autotune(
    configs=[
        # triton.Config({
        #     'BLOCK_SIZE_M': 128,
        #     'BLOCK_SIZE_N': 128,
        #     'BLOCK_SIZE_K': 32,
        #     'NUM_SM': 84,
        # }),
        # triton.Config({
        #     'BLOCK_SIZE_M': 128,
        #     'BLOCK_SIZE_N': 128,
        #     'BLOCK_SIZE_K': 32,
        #     'NUM_SM': 128,
        # }),
        # triton.Config({
        #     'BLOCK_SIZE_M': 64,
        #     'BLOCK_SIZE_N': 64,
        #     'BLOCK_SIZE_K': 32,
        #     'NUM_SM': 84,
        # }),
        triton.Config({
            'BLOCK_SIZE_M': 32,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128,
        }, num_warps=2, num_stages=5),
    ],
    key=['group_size'],
)
@triton.jit
def grouped_matmul_kernel(
    # device tensor of matrices pointers
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
    # number of virtual SM
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
        # get the gemm size of the current problem
        a_offset = tl.load(cum_input_group_range + g)
        gm = tl.load(cum_input_group_range + g + 1) - a_offset
        gn = n
        gk = k
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        # iterate through the tiles in the current gemm problem
        while (tile_idx >= last_problem_end
               and tile_idx < last_problem_end + num_tiles):

            # pick up a tile from the current gemm problem
            k = gk
            a_ptr = fused_input_ptr + a_offset * lda
            b_ptr = fused_b_ptr + g * k * n
            c_ptr = fused_output_ptr + a_offset * ldc
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # do regular gemm here
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N),
                                   dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                # hint to Triton compiler to do proper loop pipelining
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])

                a = tl.load(a_ptrs,
                            mask=offs_k[None, :] < k - kk * BLOCK_SIZE_K,
                            other=0.0)
                b = tl.load(b_ptrs,
                            mask=offs_k[:, None] < k - kk * BLOCK_SIZE_K,
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

            # go to the next tile by advancing NUM_SM
            tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


def grouped_matmul(fused_input: torch.Tensor,
                   cum_group_range: torch.Tensor,
                   fused_group_b: torch.Tensor,
                   activation: str = ""):
    device = torch.device('cuda')
    assert cum_group_range.shape[0] == fused_group_b.shape[0] + 1
    group_size = cum_group_range.shape[0] - 1
    output = torch.zeros(fused_input.shape[0],
                         fused_group_b.shape[2],
                         device=device,
                         dtype=fused_input.dtype)

    # we use a fixed number of CTA, and it's auto-tunable
    grid = lambda META: (META['NUM_SM'], )
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
