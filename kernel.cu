#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <ATen/ATen.h>
#include <THC/THCAtomics.h>

#define DISPATCH_CASE_FLOATIG_TYPES(...)            \
    AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)    \
    AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)     \
    AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)            \
    AT_DISPATCH_SWITCH(
        TYPE, NAME, DISPATCH_CASE_FLOATIG_TYPES(__VA_ARGS__))

#pragma once
#ifndef USE_ROCM
  #define MOE_LDG(arg) __ldg(arg)
#else
    #define MOE_LDG(arg) *(arg)
#endif

#ifndef USE_ROCM
    #define MOE_SHFL_XOR_SYNC(var, lane_mask) __shfl_xor_sync(uint32_t(-1), var, lane_mask)
#else
    #define MOE_SHFL_XOR_SYNC(var, lane_mask) __shfl_xor(var, lane_mask)
#endif

#ifndef USE_ROCM
    #define MOE_SHFL_SYNC(var, src_lane) __shfl_sync(uint32_t(-1), var, src_lane)
#else
    #define MOE_SHFL_SYNC(var, src_lane) __shfl(var, src_lane)
#endif

#ifndef USE_ROCM
    #define MOE_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL)  \
    cudaFuncSetAttribute(FUNC, cudaFuncAttributeMaxDynamicSharedMemorySize, VAL)
#else
    #define MOE_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL)  \
    hipFuncSetAttribute(FUNC, hipFuncAttributeMaxDynamicSharedMemorySize, VAL)
#endif

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

namespace blocksparsemoe {

template<typename scalar_t>
__global__ void bincount_kernel(scalar_t *__restrict__ src, int32_t *out, size_t numel) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    for (ptrdiff_t i = index; i < numel; i += stride) {
        atomicAdd(out + (ptrdiff_t)src[i], 1)
    }
}
}

void moe_bincount(torch::Tensor src, torch::Tensor out) {
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_ALL_TYPES(
        src.scalar_type(), "bincount_kernel", [&] {
            blocksparsemoe::bincount_kernel<scalar_t><<<BLOCKS(src.numel()), THREADS, 0, stream>>>(
                src.data_ptr<scalar_t>(),
                out.data_ptr<int32_t>(),
                src.numel());
        });
}