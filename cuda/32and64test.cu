#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <random>

#define N 100000000       // 数据量
#define BLOCK_SIZE 256    // 线程块大小
#define ITERATIONS 1000   // 每个线程的计算次数
#define REPEAT 3          // 测量重复次数

// 核函数：int32 加法（防止优化）
__global__ void int32_add(const int* __restrict__ a, const int* __restrict__ b, int* __restrict__ c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        volatile int sum = 0;
        int va = __ldg(&a[idx]);
        int vb = __ldg(&b[idx]);
        for (int i = 0; i < ITERATIONS; i++) {
            sum += va + vb;
        }
        c[idx] = sum;
    }
}

// 核函数：int64 加法（防止优化）
__global__ void int64_add(const long long* __restrict__ a, const long long* __restrict__ b, long long* __restrict__ c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        volatile long long sum = 0;
        long long va = __ldg(&a[idx]);
        long long vb = __ldg(&b[idx]);
        for (int i = 0; i < ITERATIONS; i++) {
            sum += va + vb;
        }
        c[idx] = sum;
    }
}

// 核函数：float 加法（防止优化）
__global__ void float_add(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        volatile float sum = 0;
        float va = __ldg(&a[idx]);
        float vb = __ldg(&b[idx]);
        for (int i = 0; i < ITERATIONS; i++) {
            sum += va + vb;
        }
        c[idx] = sum;
    }
}

// 核函数：double 加法（防止优化）
__global__ void double_add(const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        volatile double sum = 0;
        double va = __ldg(&a[idx]);
        double vb = __ldg(&b[idx]);
        for (int i = 0; i < ITERATIONS; i++) {
            sum += va + vb;
        }
        c[idx] = sum;
    }
}

// 测量核函数时间（预热 + 重复平均）
template <typename Func, typename... Args>
float measure_kernel(Func kernel, Args... args) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热
    kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(args...);
    cudaDeviceSynchronize();

    float total_ms = 0;
    for (int i = 0; i < REPEAT; i++) {
        cudaEventRecord(start);
        kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(args...);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return total_ms / REPEAT;
}

int main() {
    // 初始化随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> int_dist(1, 100);
    std::uniform_real_distribution<float> float_dist(1.0f, 100.0f);

    // 初始化 host 数据
    std::vector<int> h_a_int32(N), h_b_int32(N);
    std::vector<long long> h_a_int64(N), h_b_int64(N);
    std::vector<float> h_a_float(N), h_b_float(N);
    std::vector<double> h_a_double(N), h_b_double(N);

    for (int i = 0; i < N; i++) {
        int val1 = int_dist(gen);
        int val2 = int_dist(gen);

        h_a_int32[i] = val1;
        h_b_int32[i] = val2;

        h_a_int64[i] = static_cast<long long>(val1);
        h_b_int64[i] = static_cast<long long>(val2);

        h_a_float[i] = static_cast<float>(val1);
        h_b_float[i] = static_cast<float>(val2);

        h_a_double[i] = static_cast<double>(val1);
        h_b_double[i] = static_cast<double>(val2);
    }

    // 分配 device 内存
    int *d_a_int32, *d_b_int32, *d_c_int32;
    long long *d_a_int64, *d_b_int64, *d_c_int64;
    float *d_a_float, *d_b_float, *d_c_float;
    double *d_a_double, *d_b_double, *d_c_double;

    cudaMalloc(&d_a_int32, N * sizeof(int));
    cudaMalloc(&d_b_int32, N * sizeof(int));
    cudaMalloc(&d_c_int32, N * sizeof(int));

    cudaMalloc(&d_a_int64, N * sizeof(long long));
    cudaMalloc(&d_b_int64, N * sizeof(long long));
    cudaMalloc(&d_c_int64, N * sizeof(long long));

    cudaMalloc(&d_a_float, N * sizeof(float));
    cudaMalloc(&d_b_float, N * sizeof(float));
    cudaMalloc(&d_c_float, N * sizeof(float));

    cudaMalloc(&d_a_double, N * sizeof(double));
    cudaMalloc(&d_b_double, N * sizeof(double));
    cudaMalloc(&d_c_double, N * sizeof(double));

    // 复制数据到 device
    cudaMemcpy(d_a_int32, h_a_int32.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_int32, h_b_int32.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_a_int64, h_a_int64.data(), N * sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_int64, h_b_int64.data(), N * sizeof(long long), cudaMemcpyHostToDevice);

    cudaMemcpy(d_a_float, h_a_float.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_float, h_b_float.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_a_double, h_a_double.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_double, h_b_double.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // 测试 int32
    float time_int32 = measure_kernel(int32_add, d_a_int32, d_b_int32, d_c_int32);
    float giops_int32 = ((long long)N * ITERATIONS) / (time_int32 * 1e6);
    printf("int32 性能: %.2f GIOPS, 平均耗时: %.2f ms\n", giops_int32, time_int32);

    // 测试 int64
    float time_int64 = measure_kernel(int64_add, d_a_int64, d_b_int64, d_c_int64);
    float giops_int64 = ((long long)N * ITERATIONS) / (time_int64 * 1e6);
    printf("int64 性能: %.2f GIOPS, 平均耗时: %.2f ms\n", giops_int64, time_int64);

    // 测试 float
    float time_float = measure_kernel(float_add, d_a_float, d_b_float, d_c_float);
    float gflops_float = ((long long)N * ITERATIONS) / (time_float * 1e6);
    printf("float 性能: %.2f GFLOPS, 平均耗时: %.2f ms\n", gflops_float, time_float);

    // 测试 double
    float time_double = measure_kernel(double_add, d_a_double, d_b_double, d_c_double);
    float gflops_double = ((long long)N * ITERATIONS) / (time_double * 1e6);
    printf("double 性能: %.2f GFLOPS, 平均耗时: %.2f ms\n", gflops_double, time_double);

    // 释放 device 内存
    cudaFree(d_a_int32);
    cudaFree(d_b_int32);
    cudaFree(d_c_int32);
    cudaFree(d_a_int64);
    cudaFree(d_b_int64);
    cudaFree(d_c_int64);
    cudaFree(d_a_float);
    cudaFree(d_b_float);
    cudaFree(d_c_float);
    cudaFree(d_a_double);
    cudaFree(d_b_double);
    cudaFree(d_c_double);

    return 0;
}
