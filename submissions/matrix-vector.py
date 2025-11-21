import torch
import triton
import triton.language as tl

# Define block size
BLOCK_SIZE_M = 32  # Optimized for vector operations

@triton.jit
def matvec_kernel(
    # Pointers to matrices and vectors
    a_ptr, b_ptr, c_ptr,
    # Matrix/vector strides
    stride_am, stride_ak,
    stride_b,
    stride_c,
    # Sizes
    M, K,
    # Block size as constexpr for compiler optimization
    BLOCK_SIZE_M: tl.constexpr = BLOCK_SIZE_M
):
    # Program ID - which block of output are we computing
    pid_m = tl.program_id(0)
    
    # Compute start index for this block
    start_m = pid_m * BLOCK_SIZE_M
    
    # Create offsets for each element in the block
    offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    
    # Initialize accumulator for the output elements
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Compute matrix-vector multiplication for this block
    for k in range(K):
        # Load slice from matrix A
        a = tl.load(a_ptr + offsets_m * stride_am + k * stride_ak, 
                   mask=offsets_m < M, other=0.0)
        
        # Load element from vector B
        b = tl.load(b_ptr + k * stride_b)
        
        # Multiply and accumulate
        acc += a * b
    
    # Store result
    c_ptrs = c_ptr + offsets_m * stride_c
    c_mask = offsets_m < M
    tl.store(c_ptrs, acc, mask=c_mask)

def solution(input_a: torch.Tensor, input_b: torch.Tensor, output_c: torch.Tensor, m: int, k: int):
    # Launch grid - each thread handles a block of output vector
    grid = (triton.cdiv(m, BLOCK_SIZE_M),)
    
    matvec_kernel[grid](
        input_a, input_b, output_c,
        input_a.stride(0), input_a.stride(1),
        input_b.stride(0),
        output_c.stride(0),
        m, k
    )