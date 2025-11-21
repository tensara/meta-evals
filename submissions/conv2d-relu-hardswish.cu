#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

__global__ void conv_relu_hardswish_kernel(
    const float* __restrict__ image,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    size_t H, size_t W, size_t Kh, size_t Kw) 
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < H && j < W) {
        int pad_h = (Kh - 1) / 2;
        int pad_w = (Kw - 1) / 2;

        float acc = 0.0f;
        for (int u = 0; u < Kh; u++) {
            for (int v = 0; v < Kw; v++) {
                int x = i + u - pad_h;
                int y = j + v - pad_w;
                if (x >= 0 && x < H && y >= 0 && y < W) {
                    acc += image[x * W + y] * kernel[u * Kw + v];
                }
            }
        }

        float relu = fmaxf(0.0f, acc);

        float relu6 = fminf(6.0f, fmaxf(0.0f, relu + 3.0f));
        float hswish = relu * (relu6 / 6.0f);

        output[i * W + j] = hswish;
    }
}

extern "C" void solution(
    const float* image,
    const float* kernel,
    float* output,
    size_t H, size_t W, size_t Kh, size_t Kw) 
{
    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y);

    conv_relu_hardswish_kernel<<<grid, block>>>(image, kernel, output, H, W, Kh, Kw);
    cudaDeviceSynchronize();
}
