#include <cuda_runtime.h>
#include <cmath>

// Compute L2 norm with epsilon for stability
__device__ float l2_norm(const float* a, const float* b, size_t E) {
    float sum = 0.0f;
    for (size_t i = 0; i < E; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrtf(sum + 1e-6f);
}

// Compute per-sample triplet loss and store in temporary buffer
__global__ void triplet_loss_kernel(
    const float* anchor, const float* positive, const float* negative,
    float* temp_losses, size_t B, size_t E, float margin
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B) {
        const float* a = anchor + i * E;
        const float* p = positive + i * E;
        const float* n = negative + i * E;

        float d_ap = l2_norm(a, p, E);
        float d_an = l2_norm(a, n, E);

        float triplet_loss = d_ap - d_an + margin;
        temp_losses[i] = fmaxf(triplet_loss, 0.0f);
    }
}

// Reduce per-sample losses into a single mean loss
__global__ void reduce_mean_kernel(const float* temp_losses, float* out_loss, size_t B) {
    __shared__ float shared[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (i < B) ? temp_losses[i] : 0.0f;
    shared[tid] = val;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    // Write result from block 0, thread 0
    if (tid == 0) {
        atomicAdd(out_loss, shared[0]);
    }
}

extern "C" void solution(
    const float* anchor, const float* positive, const float* negative,
    float* loss, size_t B, size_t E, float margin
) {
    // Allocate temporary buffer for per-sample losses
    float* temp_losses;
    cudaMalloc(&temp_losses, B * sizeof(float));
    cudaMemset(loss, 0, sizeof(float));  // Reset output

    int threads = 256;
    int blocks = (B + threads - 1) / threads;

    // Step 1: compute per-sample loss
    triplet_loss_kernel<<<blocks, threads>>>(anchor, positive, negative, temp_losses, B, E, margin);

    // Step 2: reduce to a single scalar (mean)
    reduce_mean_kernel<<<blocks, threads>>>(temp_losses, loss, B);

    // Step 3: divide total sum by B
    float h_loss;
    cudaMemcpy(&h_loss, loss, sizeof(float), cudaMemcpyDeviceToHost);
    h_loss /= static_cast<float>(B);
    cudaMemcpy(loss, &h_loss, sizeof(float), cudaMemcpyHostToDevice);

    // Cleanup
    cudaFree(temp_losses);
    cudaDeviceSynchronize();
}
