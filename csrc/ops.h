#pragma once

#include <cstdint>
#include <torch/extension.h>

void moe_bincount(
    torch::Tensor src,
    torch::Tensor out);