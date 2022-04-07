#include <cuda_runtime.h>
#include <cstdio>

const int N = 1 << 14;

/**
 * @brief constant memory
 * 
 */ 
__constant__ float x_constDevice[N];
__global__ void calc(/*float *A, float *x,*/ float *y) {
    // normal implementation
    // for(int i = 0; i < N; ++ i)
    //     for(int j = 0; j < N; ++ j)
    //         y[i] += A[i][j] * x[j];
    // printf("%d %d %d %d\n",blockDim.x, blockIdx.x, gridDim.x, threadIdx.x);
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int step = blockDim.x * gridDim.x;
#define A(i,j) (i - 0.1 * j + 1)
    for(int i = tid; i < N; i += step)
        for(int j = 0; j < N; ++ j)
            // y[i] += A[idA] * x[j];
            y[i] += A(i,j) * x_constDevice[j];
#undef A
}
#ifdef NDEBUG
    #define cudaWork(work) work
#else
    #define cudaWork(work) \
        do { \
            work; \
            cudaError_t err = cudaGetLastError(); \
            if(err) { \
                fprintf(stderr, "error occur in line %d, %s\n", __LINE__, cudaGetErrorString(err)); \
                exit(EXIT_FAILURE); \
            } \
        } while(0)
#endif

int main() {
    float *x_host, *y, *res;
    x_host = (float*) malloc(sizeof(float) * N);
    res = (float*) malloc(sizeof(float) * N);
    cudaWork(cudaMalloc(&y, sizeof(float) * N));
    for(int i = 0; i < N; ++ i)
        x_host[i] = log(sqrt(i * i - i + 2.0));
    cudaWork(cudaMemcpyToSymbol(x_constDevice, x_host, sizeof(float) * N));
    cudaEvent_t start, stop;
    float t = 0;
    cudaWork(cudaEventCreate(&start));
    cudaWork(cudaEventCreate(&stop));
    cudaWork(cudaEventRecord(start));
    cudaWork((calc<<<2,16>>>(y)));
    cudaWork(cudaEventRecord(stop));
    cudaWork(cudaEventSynchronize(stop));
    cudaWork(cudaEventElapsedTime(&t, start, stop));
    printf("elapsedTime %fms\n", t);
    cudaWork(cudaMemcpy(res, y, sizeof(float) * N, cudaMemcpyDeviceToHost));
    // for(int i = 0; i < N ; ++ i)
    //     printf("%f\n", res[i]);
    
    cudaFree(y);
    free(res);
    free(x_host);
    
}
