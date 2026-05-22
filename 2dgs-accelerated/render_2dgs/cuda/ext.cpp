/*
 * ext.cpp - PyBind11 bindings
 * This file exposes C++ functions to Python
 */

#include <torch/extension.h>

// Declare functions implemented in operations.cu
torch::Tensor add_tensors_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor multiply_scalar_cuda(torch::Tensor input, float scalar);
void hello_cuda();

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Simple CUDA Hello World Example";
    
    // Bind C++ functions to Python names
    m.def("add", &add_tensors_cuda, "Add two tensors on GPU");
    m.def("multiply", &multiply_scalar_cuda, "Multiply tensor by scalar on GPU");
    m.def("hello", &hello_cuda, "Print hello from CUDA");
}
