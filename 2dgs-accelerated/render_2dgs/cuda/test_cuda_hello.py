"""
Test script for cuda_hello module

Run after installing:
    cd cuda_hello_world
    pip install -e .
    python test_cuda_hello.py
"""

#### =========================================================================
#### [CLAUDE-REVIEW · 2026-05-07] — added by AI, NOT part of original file
####
#### All lines starting with "####" in this file are review notes inserted
#### by Claude. The actual test code below has NOT been modified.
####
#### Issues:
####
####   #A  test_multiply only PRINTS "Match: True/False" — it does not
####       assert. Combined with the kernel bug (operations.cu Issue #1,
####       stray "+ 1024.0"), this script claims "All tests passed!" while
####       multiply is silently wrong. Replace `print(f"Match: {...}")`
####       with `assert torch.allclose(y, expected)`.
####
####   #B  test_large_tensor: the PyTorch baseline is timed without a
####       warm-up iteration, so its first launch includes lazy CUDA
####       module loading / context init. Do `_ = a + b; torch.cuda.synchronize()`
####       before the timed loop, the same way the custom kernel is warmed.
####
#### =========================================================================

import torch
from cuda_hello import hello, add, multiply

def test_hello():
    print("=" * 50)
    print("Test 1: Hello from CUDA")
    print("=" * 50)
    hello()
    print()

def test_add():
    print("=" * 50)
    print("Test 2: Add two tensors on GPU")
    print("=" * 50)
    
    # Create tensors on GPU
    a = torch.tensor([123.0, 2.0, 3.0, 4.0, 5.0], device='cuda')
    b = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0], device='cuda')
    
    print(f"a = {a}")
    print(f"b = {b}")
    
    # Add using our CUDA kernel
    c = add(a, b)
    print(f"add(a, b) = {c}")
    
    # Verify with PyTorch
    expected = a + b
    print(f"Expected (PyTorch): {expected}")
    print(f"Match: {torch.allclose(c, expected)}")
    print()

def test_multiply():
    print("=" * 50)
    print("Test 3: Multiply tensor by scalar on GPU")
    print("=" * 50)
    
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
    scalar = 2.5
    
    print(f"x = {x}")
    print(f"scalar = {scalar}")
    
    # Multiply using our CUDA kernel
    y = multiply(x, scalar)
    print(f"multiply(x, {scalar}) = {y}")
    
    # Verify with PyTorch
    expected = x * scalar
    print(f"Expected (PyTorch): {expected}")
    #### [CLAUDE-REVIEW] Issue #A — this prints the result but does not
    #### assert. Replace with:  assert torch.allclose(y, expected)
    print(f"Match: {torch.allclose(y, expected)}")
    print()

def test_large_tensor():
    print("=" * 50)
    print("Test 4: Large tensor performance")
    print("=" * 50)
    
    N = 10_000_000
    a = torch.randn(N, device='cuda')
    b = torch.randn(N, device='cuda')
    
    print(f"Tensor size: {N:,} elements")
    
    # Warm up
    _ = add(a, b)
    torch.cuda.synchronize()
    
    # Time our CUDA kernel
    import time
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        c = add(a, b)
    torch.cuda.synchronize()
    our_time = time.time() - start
    
    #### [CLAUDE-REVIEW] Issue #B — PyTorch baseline below is not warmed
    #### up the way the custom kernel is. The first `a + b` includes lazy
    #### module load / context init, biasing pytorch_time upward. Add:
    ####     _ = a + b; torch.cuda.synchronize()
    #### before the timed loop.
    # Time PyTorch
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        c = a + b
    torch.cuda.synchronize()
    pytorch_time = time.time() - start
    
    print(f"Our CUDA kernel: {our_time*1000:.2f} ms (100 iterations)")
    print(f"PyTorch built-in: {pytorch_time*1000:.2f} ms (100 iterations)")
    print()

if __name__ == "__main__":
    print("\n🚀 Testing cuda_hello module\n")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        exit(1)
    
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}\n")
    
    test_hello()
    test_add()
    test_multiply()
    test_large_tensor()
    
    print("✅ All tests passed!")
