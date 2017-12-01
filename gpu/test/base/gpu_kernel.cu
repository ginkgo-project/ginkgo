
#ifndef THREADS_PER_PROC
#define THREADS_PER_PROC 20
#endif


#ifndef NUM_PROC
#define NUM_PROC 5
#endif

template <typename T1, typename T2>
__global__ void gpu_kernel(
        T1 size, T2 *x)
{
    const int nt = blockDim.x;
    const int np = gridDim.x;
    const int tid = threadIdx.x + blockIdx.x * nt;
    for (int i = tid; i < size; i += np*nt){
        x[i] = 0.5 * x[i] + 1.0;
    }

}

template <typename T1, typename T2>
void run_on_gpu(T1 size, T2 *d_x){
    gpu_kernel<<<1,1>>>(size,d_x);
}
