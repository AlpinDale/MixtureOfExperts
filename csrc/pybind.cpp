#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include "ops.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moe_bincount", &moe_bincount, py::arg("src"), py::arg("out"), "MOE bincount (CUDA)");
}