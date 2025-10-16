// File: cuda_fused_gemm.cu

#include <cuda_runtime.h>
#include <math.h>

// GELU function (approximate)
__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// Fused GEMM + bias + activation kernel (row-major, FP32)
__global__ void fused_gemm_bias_gelu(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        // Compute dot-product for C[row, col]
        float val = 0.0f;
        for (int k = 0; k < K; ++k) {
            val += A[row * K + k] * B[k * N + col];
        }
        // Add bias and apply GELU
        val += bias[col];
        val = gelu(val);
        C[row * N + col] = val;
    }
}

// Host-side launcher utility (can be called from C++ or with PyCUDA bindings)
extern "C"
void launch_fused_gemm_bias_gelu(
    const float* A, const float* B, const float* bias, float* C,
    int M, int N, int K,
    cudaStream_t stream
) {
    dim3 blockDim(16, 16); // You may tune this block size
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);
    fused_gemm_bias_gelu<<<gridDim, blockDim, 0, stream>>>(
        A, B, bias, C, M, N, K
    );
}
