import triton
import triton.language as tl


@triton.jit
def huber_kernel(predictions_ptr, targets_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    predictions = tl.load(predictions_ptr + offsets, mask=mask)
    targets = tl.load(targets_ptr + offsets, mask=mask)
    
    diff = predictions - targets
    abs_diff = tl.abs(diff)
    
    result = tl.where(abs_diff < 1.0, 0.5 * diff * diff, abs_diff - 0.5)
    
    tl.store(output_ptr + offsets, result, mask=mask)


def solution(predictions, targets, output, n):
    n_elements = n
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    huber_kernel[grid](predictions, targets, output, n_elements, BLOCK_SIZE=1024)

