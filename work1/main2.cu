#include <cuda_runtime.h>
#include <cstdio>

const int N = 1 << 14;
const int Dim = 32;

/**
 * @brief coalesced memory
 * no need for the optimization
 */ 

__global__ void calc();

#ifdef NDEBUG
    #define cudaWork(work) work
#else
    #define cudaWork(work) \
        do { \
            work; \
            cudaError_t err = cudaGetLastError(); \
            if(err) { \
                printf("error occur in line %d, %s\n", __LINE__, cudaGetErrorString(err)); \
                exit(EXIT_FAILURE); \
            } \
        } while(0)
#endif
int main() {
    
}
