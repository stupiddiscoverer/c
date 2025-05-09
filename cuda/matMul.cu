#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

// 检查CUDA运行时API的返回值
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// CUDA 矩阵乘法内核
__global__ void matrixMulKernel(const float* A, const float* B, float* C, 
                               int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * K + col]; // A的第row行 × B的第col列
        }
        C[row * K + col] = sum;
    }
}

// CPU矩阵乘法（用于验证）
void matrixMulCPU(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < K; ++col) {
            float sum = 0.0f;
            for (int i = 0; i < N; ++i) {
                sum += A[row * N + i] * B[i * K + col];
            }
            C[row * K + col] = sum;
        }
    }
}

// GPU矩阵乘法
void matrixMulGPU(const float* h_A, const float* h_B, float* h_C, 
                 int M, int N, int K) {
    // 1. 分配设备内存
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, N * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * K * sizeof(float)));

    // 2. 拷贝数据到设备
    CHECK_CUDA(cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, N * K * sizeof(float), cudaMemcpyHostToDevice));

    // 3. 配置线程块和网格
    dim3 blockSize(16, 16); // 256 threads per block
    dim3 gridSize((K + blockSize.x - 1) / blockSize.x, 
                 (M + blockSize.y - 1) / blockSize.y);

    // 4. 启动内核并检查错误
    matrixMulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize()); // 等待内核完成

    // 5. 拷贝结果回主机
    CHECK_CUDA(cudaMemcpy(h_C, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost));

    // 6. 释放设备内存
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
}

void printMatrix(const float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%8.2f", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("------------------------------------------\n");
}

bool verifyResult(const float* cpuResult, const float* gpuResult, int size) {
    const float epsilon = 1e-5f;
    for (int i = 0; i < size; ++i) {
        if (fabs(cpuResult[i] - gpuResult[i]) > epsilon) {
            printf("Mismatch at index %d: CPU=%.2f, GPU=%.2f\n", 
                   i, cpuResult[i], gpuResult[i]);
            return false;
        }
    }
    return true;
}

int main() {
    const int M = 4; // A的行数
    const int N = 3; // A的列数（B的行数）
    const int K = 5; // B的列数

    // 分配主机内存
    float *h_A = new float[M * N];
    float *h_B = new float[N * K];
    float *h_C_gpu = new float[M * K];
    float *h_C_cpu = new float[M * K];

    // 初始化矩阵（使用简单值便于验证）
    for (int i = 0; i < M * N; ++i) h_A[i] = i + 1; // 1, 2, 3, ...
    for (int i = 0; i < N * K; ++i) h_B[i] = i + 1; // 1, 2, 3, ...

    // 打印输入矩阵
    printf("Matrix A (%d x %d):\n", M, N);
    printMatrix(h_A, M, N);
    printf("Matrix B (%d x %d):\n", N, K);
    printMatrix(h_B, N, K);

    // 在GPU上计算
    matrixMulGPU(h_A, h_B, h_C_gpu, M, N, K);

    // 在CPU上计算（用于验证）
    matrixMulCPU(h_A, h_B, h_C_cpu, M, N, K);

    // 打印结果
    printf("GPU Result (%d x %d):\n", M, K);
    printMatrix(h_C_gpu, M, K);
    printf("CPU Result (%d x %d):\n", M, K);
    printMatrix(h_C_cpu, M, K);

    // 验证结果
    if (verifyResult(h_C_cpu, h_C_gpu, M * K)) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do NOT match!" << std::endl;
    }

    // 释放内存
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_gpu;
    delete[] h_C_cpu;

    return 0;
}