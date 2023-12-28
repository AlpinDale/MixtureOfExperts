#include <torch/extension.h>

#pragma once
void moe_bincount(
    torch::Tensor src,
    torch::Tensor out)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::module blocksparse_ops = m.def_submodule("blocksparse_ops",
                                                       "BlockSparse MoE custom ops.")
    ops.def(
        "bincount",
        &moe_bincount,
        "Gather key/value from the cache into contiguous QKV tensors.")
}