#include<stdio.h>


__global__ void hello_cu()
{
    printf("bid.x=%d, bid.y=%d, tid.x=%d\n", blockIdx.x, blockIdx.y, threadIdx.x);
}

__global__ void addArray(int *a, int *b, int *c)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < 5)
        c[id] = a[id] + b[id];
}

void gpuInfo()
{
    cudaDeviceProp cudaProp;
    cudaGetDeviceProperties(&cudaProp, 0);

    printf("device name:             %s\n", cudaProp.name);
    printf("compute capability       %d.%d\n", cudaProp.major, cudaProp.minor);
    printf("global memory:           %lu MB\n", cudaProp.totalGlobalMem >> 20);
    printf("constant memory:         %lu KB\n", cudaProp.totalConstMem >> 10);
    printf("grid size:               %d %d %d\n", cudaProp.maxGridSize[0],\
                            cudaProp.maxGridSize[1], cudaProp.maxGridSize[2]);
    printf("block size:              %d %d %d\n", cudaProp.maxThreadsDim[0],\
                            cudaProp.maxThreadsDim[1],cudaProp.maxThreadsDim[2]);
    printf("multiProcessorCount:        %d\n", cudaProp.multiProcessorCount);
    printf("maxThreadsPerMultiProcessor %d\n", cudaProp.maxThreadsPerMultiProcessor);
    printf("maxBlocksPerMultiProcessor  %d\n", cudaProp.maxBlocksPerMultiProcessor);
    printf("maxThreadsPerBlock:         %d\n", cudaProp.maxThreadsPerBlock);
    printf("accessPolicyMaxWindowSize %d\n", cudaProp.accessPolicyMaxWindowSize);
    printf("memoryBusWidth           %d\n", cudaProp.memoryBusWidth);
    printf("memoryClockRate          %d\n", cudaProp.memoryClockRate);
    printf("managedMemory            %d\n", cudaProp.managedMemory);
    printf("canMapHostMemory         %d\n", cudaProp.canMapHostMemory);
    printf("accessPolicyMaxWindowSize %d\n", cudaProp.accessPolicyMaxWindowSize);
    printf("asyncEngineCount         %d\n", cudaProp.asyncEngineCount);
    
    printf("FP64 support: %s, INT64 support: Yes\n", (cudaProp.major >= 2) ? "Yes (但性能低)" : "No");
}

void addTest()
{
    int a[] = {1,2,3,4,5};
    int *b, *c, *d;
    cudaMalloc(&b, (size_t)20);
    cudaMalloc(&c, (size_t)20);
    cudaMalloc(&d, (size_t)20);
    cudaMemset(b, 0, 20);
    cudaMemcpy(b, a, 3*4, cudaMemcpyHostToDevice);
    cudaMemcpy(c, a, 5*4, cudaMemcpyHostToDevice);

    dim3 block(32);
    dim3 grid(1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    addArray<<<block, grid>>>(b,c,d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("using time %f ms\n", elapsedTime);

    cudaMemcpy(a, d, 5*4, cudaMemcpyDeviceToHost);

    for(int i=0; i<5; i++)
    {
        printf("a[%d] = %d\n", i, a[i]);
    }
    cudaFree(b);
    cudaFree(c);
    cudaFree(d);
    cudaDeviceReset();
}

int main(int argc, char const *argv[])
{
    hello_cu<<<2, 3>>>();
    printf("hello\n");
    cudaDeviceSynchronize();

    int devices = 0;
    cudaGetDeviceCount(&devices);
    printf("devices = %d\n", devices);
    cudaError_t dick = cudaSetDevice(0);   //调用gpu：0
    printf("cudaSuccess = %d, dick = %d\n", cudaSuccess, dick);

    gpuInfo();
    return 0;
}
