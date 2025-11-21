import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(A_ptr, B_ptr, C_ptr, N, K, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    offset = (K - 1) // 2
    result = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for j in range(K):
        input_offsets = offsets + (j - offset)
        input_mask = (input_offsets >= 0) & (input_offsets < N) & mask
        input_vals = tl.load(A_ptr + input_offsets, mask=input_mask, other=0.0)
        kernel_val = tl.load(B_ptr + j)
        result += input_vals * kernel_val
    
    tl.store(C_ptr + offsets, result, mask=mask)


def solution(A, B, C, N, K):
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    conv1d_kernel[grid](A, B, C, N, K, BLOCK_SIZE=1024)

