#include <cuda_runtime.h>
#include <math.h>

__global__ void huberLossKernel(const float* predictions, const float* targets, float* output, size_t n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < n) {
        float diff = predictions[idx] - targets[idx];
        float abs_diff = fabsf(diff);
        
        if (abs_diff < 1.0f) {
            output[idx] = 0.5f * diff * diff;
        } else {
            output[idx] = abs_diff - 0.5f;
        }
    }
}

extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n) {
    const int threadsPerBlock = 256;
    const int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    huberLossKernel<<<numBlocks, threadsPerBlock>>>(predictions, targets, output, n);
}

