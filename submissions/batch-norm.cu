#include <cuda_runtime.h>
#include <stdio.h>

__global__ void compute_mean_kernel(
    const float* __restrict__ X,
    float* __restrict__ means,
    const size_t B,
    const size_t F,
    const size_t D1,
    const size_t D2
) {
    const int f = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const size_t spatial_size = D1 * D2;
    const size_t feature_size = F * spatial_size;
    
    __shared__ float shared_sum[256];
    shared_sum[tid] = 0.0f;
    
    float thread_sum = 0.0f;
    for (size_t b = 0; b < B; b++) {
        for (size_t d1 = 0; d1 < D1; d1++) {
            for (size_t d2 = tid; d2 < D2; d2 += stride) {
                const size_t idx = b * feature_size + f * spatial_size + d1 * D2 + d2;
                thread_sum += X[idx];
            }
        }
    }
    
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        means[f] = shared_sum[0] / static_cast<float>(B * D1 * D2);
    }
}

__global__ void compute_variance_kernel(
    const float* __restrict__ X,
    const float* __restrict__ means,
    float* __restrict__ variances,
    const size_t B,
    const size_t F,
    const size_t D1,
    const size_t D2
) {
    const int f = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const size_t spatial_size = D1 * D2;
    const size_t feature_size = F * spatial_size;
    const float mean = means[f];
    
    __shared__ float shared_sum[256];
    shared_sum[tid] = 0.0f;
    
    float thread_sum = 0.0f;
    for (size_t b = 0; b < B; b++) {
        for (size_t d1 = 0; d1 < D1; d1++) {
            for (size_t d2 = tid; d2 < D2; d2 += stride) {
                const size_t idx = b * feature_size + f * spatial_size + d1 * D2 + d2;
                const float diff = X[idx] - mean;
                thread_sum += diff * diff;
            }
        }
    }
    
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        variances[f] = shared_sum[0] / static_cast<float>(B * D1 * D2);
    }
}

__global__ void normalize_kernel(
    const float* __restrict__ X,
    float* __restrict__ Y,
    const float* __restrict__ means,
    const float* __restrict__ variances,
    const float epsilon,
    const size_t B,
    const size_t F,
    const size_t D1,
    const size_t D2
) {
    const size_t b = blockIdx.z;
    const size_t f = blockIdx.y;
    const size_t d1 = blockIdx.x;
    const size_t tid = threadIdx.x;
    const size_t stride = blockDim.x;
    const size_t spatial_size = D1 * D2;
    const size_t feature_size = F * spatial_size;
    
    const float mean = means[f];
    const float inv_std = 1.0f / sqrtf(variances[f] + epsilon);
    
    for (size_t d2 = tid; d2 < D2; d2 += stride) {
        const size_t idx = b * feature_size + f * spatial_size + d1 * D2 + d2;
        Y[idx] = (X[idx] - mean) * inv_std;
    }
}

extern "C" void solution(const float* X, float* Y, size_t B, size_t F, size_t D1, size_t D2) {
    float *d_means, *d_variances;
    cudaMalloc(&d_means, F * sizeof(float));
    cudaMalloc(&d_variances, F * sizeof(float));
    
    const int block_size = 256;
    compute_mean_kernel<<<F, block_size>>>(X, d_means, B, F, D1, D2);
    
    compute_variance_kernel<<<F, block_size>>>(X, d_means, d_variances, B, F, D1, D2);
    
    const float epsilon = 1e-5f;
    dim3 normalize_grid(D1, F, B);
    normalize_kernel<<<normalize_grid, block_size>>>(X, Y, d_means, d_variances, epsilon, B, F, D1, D2);

    cudaFree(d_means);
    cudaFree(d_variances);
}