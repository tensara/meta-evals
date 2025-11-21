import triton
import triton.language as tl

@triton.jit
def leaky_relu_kernel(input_ptr, output_ptr, n, m, alpha, BLOCK_SIZE: tl.constexpr):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    offsets_x = tl.arange(0, BLOCK_SIZE)
    offsets_y = tl.arange(0, BLOCK_SIZE)
    
    x = pid_x * BLOCK_SIZE + offsets_x
    y = pid_y * BLOCK_SIZE + offsets_y
    
    mask_x = x < n
    mask_y = y < m
    mask = mask_x[:, None] & mask_y[None, :]
    
    index = y[:, None] * n + x[None, :]
    
    input_data = tl.load(input_ptr + index, mask=mask, other=0.0)
    
    output_data = tl.where(input_data >= 0, input_data, alpha * input_data)
    
    tl.store(output_ptr + index, output_data, mask=mask)

def solution(input, alpha: float, output, n: int, m: int):
    BLOCK_SIZE = 16
    grid_x = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_y = (m + BLOCK_SIZE - 1) // BLOCK_SIZE

    leaky_relu_kernel[(grid_x, grid_y)](input, output, n, m, alpha, BLOCK_SIZE)
