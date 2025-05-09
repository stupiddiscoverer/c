// nvcc fc5_test.cu -lcublas -o fc5_test
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call)  { const cudaError_t e = call; if (e != cudaSuccess) { printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } }
#define CHECK_CUBLAS(call) { const cublasStatus_t s = call; if (s != CUBLAS_STATUS_SUCCESS) { printf("cuBLAS error %s:%d\n", __FILE__, __LINE__); exit(1); } }

const int batch_size = 64;
const int dim_in = 4096;
const int dim_hidden = 4096;
// const int dim_out = 4096;
const int layers = 5;

__global__ void relu(float* x, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) x[i] = fmaxf(0.0f, x[i]);
}

__global__ void relu_half(__half* x, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) x[i] = __hgt(x[i], __float2half(0.0f)) ? x[i] : __float2half(0.0f);
}

void fc5_cublas_fp32(cublasHandle_t handle, float* input, float** weights, float** biases, float* output) {
    const float alpha = 1.0f, beta = 1.0f;
    float *x = input, *y;
    int M = batch_size, N = dim_hidden, K = dim_in;
    for (int l = 0; l < layers; l++) {
        y = output + l * M * N;  // 每层结果单独存
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, weights[l], N, x, K, &beta, y, N));
        CHECK_CUDA(cudaMemcpy(y, biases[l], sizeof(float)*N, cudaMemcpyDeviceToDevice));  // 加 bias
        int size = M * N;
        relu<<<(size+255)/256, 256>>>(y, size);
        x = y;
        K = N;
    }
}

void fc5_cublas_fp16(cublasHandle_t handle, __half* input, __half** weights, __half** biases, __half* output) {
    const __half alpha = __float2half(1.0f), beta = __float2half(1.0f);
    __half *x = input, *y;
    int M = batch_size, N = dim_hidden, K = dim_in;
    for (int l = 0; l < layers; l++) {
        y = output + l * M * N;
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
            &alpha, weights[l], CUDA_R_16F, N,
                    x, CUDA_R_16F, K,
            &beta,  y, CUDA_R_16F, N,
            CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CHECK_CUDA(cudaMemcpy(y, biases[l], sizeof(__half)*N, cudaMemcpyDeviceToDevice));
        int size = M * N;
        relu_half<<<(size+255)/256, 256>>>(y, size);
        x = y;
        K = N;
    }
}

int main() {
    printf("Allocating...\n");
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    size_t size_f32 = batch_size * dim_hidden * layers * sizeof(float);
    size_t size_f16 = batch_size * dim_hidden * layers * sizeof(__half);

    // FP32 alloc
    float *input32, *output32, *weights32[layers], *bias32[layers];
    CHECK_CUDA(cudaMalloc(&input32, batch_size * dim_in * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&output32, size_f32));
    for (int l=0; l<layers; l++) {
        CHECK_CUDA(cudaMalloc(&weights32[l], dim_hidden * dim_in * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&bias32[l], dim_hidden * sizeof(float)));
    }

    // FP16 alloc
    __half *input16, *output16, *weights16[layers], *bias16[layers];
    CHECK_CUDA(cudaMalloc(&input16, batch_size * dim_in * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&output16, size_f16));
    for (int l=0; l<layers; l++) {
        CHECK_CUDA(cudaMalloc(&weights16[l], dim_hidden * dim_in * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&bias16[l], dim_hidden * sizeof(__half)));
    }

    printf("Warmup...\n");
    fc5_cublas_fp32(handle, input32, weights32, bias32, output32);
    fc5_cublas_fp16(handle, input16, weights16, bias16, output16);
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("Measuring FP32...\n");
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i=0; i<10; i++)
        fc5_cublas_fp32(handle, input32, weights32, bias32, output32);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_fp32;
    cudaEventElapsedTime(&ms_fp32, start, stop);

    printf("Measuring FP16 Tensor Core...\n");
    CHECK_CUDA(cudaEventRecord(start));
    for (int i=0; i<10; i++)
        fc5_cublas_fp16(handle, input16, weights16, bias16, output16);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_fp16;
    cudaEventElapsedTime(&ms_fp16, start, stop);

    printf("FP32 time: %.3f ms\n", ms_fp32);
    printf("FP16 Tensor Core time: %.3f ms\n", ms_fp16);

    // Clean up
    CHECK_CUBLAS(cublasDestroy(handle));
    cudaFree(input32); cudaFree(output32);
    cudaFree(input16); cudaFree(output16);
    for (int l=0; l<layers; l++) {
        cudaFree(weights32[l]); cudaFree(bias32[l]);
        cudaFree(weights16[l]); cudaFree(bias16[l]);
    }
    return 0;
}
