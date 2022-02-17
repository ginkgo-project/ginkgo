#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "symbolic.h"
#include <cmath>


using namespace std;


__global__ void RL(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        const unsigned* __restrict__ csr_r_ptr_dev,
        const unsigned* __restrict__ csr_c_idx_dev,
        const unsigned* __restrict__ csr_diag_ptr_dev,
        const int* __restrict__ level_idx_dev,
        REAL* __restrict__ tmpMem,
        const unsigned n,
        const int levelHead,
        const int inLevPos)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int wid = threadIdx.x / 32;

    const unsigned currentCol = level_idx_dev[levelHead+inLevPos+bid];
    const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
    const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

    extern __shared__ REAL s[];

    //update current col

    int offset = 0;
    while (currentLColSize > tid + offset)
    {
        unsigned ridx = sym_r_idx_dev[currentLPos + offset];

        val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
        tmpMem[bid*n + ridx]= val_dev[currentLPos + offset];

        offset += blockDim.x;
    }
    __syncthreads();

    //broadcast to submatrix
    const unsigned subColPos = csr_diag_ptr_dev[currentCol] + wid + 1;
    const unsigned subMatSize = csr_r_ptr_dev[currentCol + 1] - csr_diag_ptr_dev[currentCol] - 1;
    unsigned subCol;
    const int tidInWarp = threadIdx.x % 32;
    unsigned subColElem = 0;

    int woffset = 0;
    while (subMatSize > woffset)
    {
        if (wid + woffset < subMatSize)
        {
            offset = 0;
            subCol = csr_c_idx_dev[subColPos + woffset];
            while(offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
            {
                if (tidInWarp + offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
                {

                    subColElem = sym_c_ptr_dev[subCol] + tidInWarp + offset;
                    unsigned ridx = sym_r_idx_dev[subColElem];

                    if (ridx == currentCol)
                    {
                        s[wid] = val_dev[subColElem];
                    }
                    //Threads in a warp are always synchronized
                    //__syncthreads();
                    if (ridx > currentCol)
                    {
                        //elem in currentCol same row with subColElem might be 0, so
                        //clearing tmpMem is necessary
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 600)
                        atomicAdd(&val_dev[subColElem], -tmpMem[ridx+n*bid]*s[wid]);
#endif
                    }
                }
                offset += 32;
            }
        }
        woffset += blockDim.x/32;
    }

    __syncthreads();
    //Clear tmpMem
    offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];
            tmpMem[bid*n + ridx]= 0;
        }
        offset += blockDim.x;
    }
}

__global__ void RL_perturb(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        const unsigned* __restrict__ csr_r_ptr_dev,
        const unsigned* __restrict__ csr_c_idx_dev,
        const unsigned* __restrict__ csr_diag_ptr_dev,
        const int* __restrict__ level_idx_dev,
        REAL* __restrict__ tmpMem,
        const unsigned n,
        const int levelHead,
        const int inLevPos,
        const float pert)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int wid = threadIdx.x / 32;

    const unsigned currentCol = level_idx_dev[levelHead+inLevPos+bid];
    const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
    const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

    extern __shared__ REAL s[];

    //update current col

    int offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];

            if (abs(val_dev[l_col_ptr_dev[currentCol]]) < pert)
                val_dev[l_col_ptr_dev[currentCol]] = pert;

            val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
            tmpMem[bid*n + ridx]= val_dev[currentLPos + offset];
        }
        offset += blockDim.x;
    }
    __syncthreads();

    //broadcast to submatrix
    const unsigned subColPos = csr_diag_ptr_dev[currentCol] + wid + 1;
    const unsigned subMatSize = csr_r_ptr_dev[currentCol + 1] - csr_diag_ptr_dev[currentCol] - 1;
    unsigned subCol;
    const int tidInWarp = threadIdx.x % 32;
    unsigned subColElem = 0;

    int woffset = 0;
    while (subMatSize > woffset)
    {
        if (wid + woffset < subMatSize)
        {
            offset = 0;
            subCol = csr_c_idx_dev[subColPos + woffset];
            while(offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
            {
                if (tidInWarp + offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
                {

                    subColElem = sym_c_ptr_dev[subCol] + tidInWarp + offset;
                    unsigned ridx = sym_r_idx_dev[subColElem];

                    if (ridx == currentCol)
                    {
                        s[wid] = val_dev[subColElem];
                    }
                    //Threads in a warp are always synchronized
                    //__syncthreads();
                    if (ridx > currentCol)
                    {
                        //elem in currentCol same row with subColElem might be 0, so
                        //clearing tmpMem is necessary
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 600)
                        atomicAdd(&val_dev[subColElem], -tmpMem[ridx+n*bid]*s[wid]);
#endif
                    }
                }
                offset += 32;
            }
        }
        woffset += blockDim.x/32;
    }

    __syncthreads();
    //Clear tmpMem
    offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];
            tmpMem[bid*n + ridx]= 0;
        }
        offset += blockDim.x;
    }
}

__global__ void RL_onecol_factorizeCurrentCol(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        const unsigned currentCol,
        REAL* __restrict__ tmpMem,
        const int stream,
        const unsigned n)
{
    const int tid = threadIdx.x;

    const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
    const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

    //update current col

    int offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];

            val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
            tmpMem[stream * n + ridx]= val_dev[currentLPos + offset];
        }
        offset += blockDim.x;
    }
}

__global__ void RL_onecol_factorizeCurrentCol_perturb(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        const unsigned currentCol,
        REAL* __restrict__ tmpMem,
        const int stream,
        const unsigned n,
        const float pert)
{
    const int tid = threadIdx.x;

    const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
    const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

    //update current col

    int offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];

            if (abs(val_dev[l_col_ptr_dev[currentCol]]) < pert)
                val_dev[l_col_ptr_dev[currentCol]] = pert;

            val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
            tmpMem[stream * n + ridx]= val_dev[currentLPos + offset];
        }
        offset += blockDim.x;
    }
}

__global__ void RL_onecol_updateSubmat(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ csr_c_idx_dev,
        const unsigned* __restrict__ csr_diag_ptr_dev,
        const unsigned currentCol,
        REAL* __restrict__ tmpMem,
        const int stream,
        const unsigned n)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    __shared__ REAL s;

    //broadcast to submatrix
    const unsigned subColPos = csr_diag_ptr_dev[currentCol] + bid + 1;
    unsigned subCol;
    unsigned subColElem = 0;

    int offset = 0;
    subCol = csr_c_idx_dev[subColPos];
    while(offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
    {
        if (tid + offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
        {
            subColElem = sym_c_ptr_dev[subCol] + tid + offset;
            unsigned ridx = sym_r_idx_dev[subColElem];

            if (ridx == currentCol)
            {
                s = val_dev[subColElem];
            }
            __syncthreads();
            if (ridx > currentCol)
            {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 600)
                atomicAdd(&val_dev[subColElem], -tmpMem[stream * n + ridx] * s);
#endif
            }
        }
        offset += blockDim.x;
    }
}

__global__ void RL_onecol_cleartmpMem(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        const unsigned currentCol,
        REAL* __restrict__ tmpMem,
        const int stream,
        const unsigned n)
{
    const int tid = threadIdx.x;

    const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
    const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

    unsigned offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];
            tmpMem[stream * n + ridx]= 0;
        }
        offset += blockDim.x;
    }
}

void LUonDevice(Symbolic_Matrix &A_sym, ostream &out, ostream &err, bool PERTURB)
{
    int deviceCount, dev;
    cudaGetDeviceCount(&deviceCount);
    cudaDeviceProp deviceProp;
    dev = 0;
    cudaGetDeviceProperties(&deviceProp, dev);
    cudaSetDevice(dev);
    out << "Device " << dev << ": " << deviceProp.name << " has been selected." << endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time;
    unsigned n = A_sym.n;
    unsigned nnz = A_sym.nnz;
    unsigned num_lev = A_sym.num_lev;
    unsigned *sym_c_ptr_dev, *sym_r_idx_dev, *l_col_ptr_dev;
    REAL *val_dev;
    unsigned *csr_r_ptr_dev, *csr_c_idx_dev, *csr_diag_ptr_dev;
    int *level_idx_dev;

    cudaEventRecord(start, 0);

    cudaMalloc((void**)&sym_c_ptr_dev, (n + 1) * sizeof(unsigned));
    cudaMalloc((void**)&sym_r_idx_dev, nnz * sizeof(unsigned));
    cudaMalloc((void**)&val_dev, nnz * sizeof(REAL));
    cudaMalloc((void**)&l_col_ptr_dev, n * sizeof(unsigned));
    cudaMalloc((void**)&csr_r_ptr_dev, (n + 1) * sizeof(unsigned));
    cudaMalloc((void**)&csr_c_idx_dev, nnz * sizeof(unsigned));
    cudaMalloc((void**)&csr_diag_ptr_dev, n * sizeof(unsigned));
    cudaMalloc((void**)&level_idx_dev, n * sizeof(int));

    cudaMemcpy(sym_c_ptr_dev, &(A_sym.sym_c_ptr[0]), (n + 1) * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(sym_r_idx_dev, &(A_sym.sym_r_idx[0]), nnz * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(val_dev, &(A_sym.val[0]), nnz * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(l_col_ptr_dev, &(A_sym.l_col_ptr[0]), n * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(csr_r_ptr_dev, &(A_sym.csr_r_ptr[0]), (n + 1) * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(csr_c_idx_dev, &(A_sym.csr_c_idx[0]), nnz * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(csr_diag_ptr_dev, &(A_sym.csr_diag_ptr[0]), n * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(level_idx_dev, &(A_sym.level_idx[0]), n * sizeof(int), cudaMemcpyHostToDevice);

    REAL* tmpMem;
    unsigned TMPMEMNUM;
    size_t MaxtmpMemSize;
    {
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        //Leave at least 4GB free, smaller size does not work for unknown reason
        MaxtmpMemSize = free - 4ull * 1024ull * 1024ull * 1024ull;
    }
    // Use size of first level to estimate a good tmpMem size
    const size_t GoodtmpMemChoice = sizeof(REAL) * size_t(n) * size_t(A_sym.level_ptr[1]);
    if (GoodtmpMemChoice < MaxtmpMemSize)
        TMPMEMNUM = A_sym.level_ptr[1];
    else {
        TMPMEMNUM = MaxtmpMemSize / n / sizeof(REAL);
    }

    const size_t tmpMemChoice = sizeof(REAL) * size_t(n) * size_t(TMPMEMNUM);
    cudaMalloc((void**)&tmpMem, tmpMemChoice);
    cudaMemset(tmpMem, 0, tmpMemChoice);

    int Nstreams = 16;
    cudaStream_t streams[Nstreams];
    for (int j = 0; j < Nstreams; ++j)
        cudaStreamCreate(&streams[j]);

    // calculate 1-norm of A and perturbation value for perturbation
    float pert = 0;
    if (PERTURB)
    {
        float norm_A = 0;
        for (int i = 0; i < n; ++i)
        {
            float tmp = 0;
            for (int j = A_sym.sym_c_ptr[i]; j < A_sym.sym_c_ptr[i+1]; ++j)
                tmp += abs(A_sym.val[j]);
            if (norm_A < tmp)
                norm_A = tmp;
        }
        pert = 1e-8 * norm_A;
        out << "Gaussian elimination with static pivoting (GESP)..." << endl;
        out << "1-Norm of A matrix is " << norm_A << ", Perturbation value is " << pert << endl;
    }


    for (int i = 0; i < num_lev; ++i)
    {
        int lev_size = A_sym.level_ptr[i + 1] - A_sym.level_ptr[i];

        if (lev_size > 896) { //3584 / 4
            //std::cout << "LEVEL " << i << " IS LARGE: " << lev_size << std::endl;
            unsigned WarpsPerBlock = 2;
            dim3 dimBlock(WarpsPerBlock * 32, 1);
            size_t MemSize = WarpsPerBlock * sizeof(REAL);

            unsigned j = 0;
            while(lev_size > 0) {
                unsigned restCol = lev_size > TMPMEMNUM ? TMPMEMNUM : lev_size;
                dim3 dimGrid(restCol, 1);
                if (!PERTURB)
                    RL<<<dimGrid, dimBlock, MemSize>>>(sym_c_ptr_dev,
                                        sym_r_idx_dev,
                                        val_dev,
                                        l_col_ptr_dev,
                                        csr_r_ptr_dev,
                                        csr_c_idx_dev,
                                        csr_diag_ptr_dev,
                                        level_idx_dev,
                                        tmpMem,
                                        n,
                                        A_sym.level_ptr[i],
                                        j*TMPMEMNUM);
                else
                    RL_perturb<<<dimGrid, dimBlock, MemSize>>>(sym_c_ptr_dev,
                                        sym_r_idx_dev,
                                        val_dev,
                                        l_col_ptr_dev,
                                        csr_r_ptr_dev,
                                        csr_c_idx_dev,
                                        csr_diag_ptr_dev,
                                        level_idx_dev,
                                        tmpMem,
                                        n,
                                        A_sym.level_ptr[i],
                                        j*TMPMEMNUM,
                                        pert);
                j++;
                lev_size -= TMPMEMNUM;

                cudaDeviceSynchronize();
                cudaError_t cudaRet = cudaGetLastError();
                if (cudaRet != cudaSuccess) {
                    out << cudaGetErrorName(cudaRet) << endl;
                    out << cudaGetErrorString(cudaRet) << endl;
                }
            }
        }
        else if (lev_size > 448) {
            //std::cout << "LEVEL " << i << " IS MEDIUM: " << lev_size << std::endl;
            unsigned WarpsPerBlock = 4;
            dim3 dimBlock(WarpsPerBlock * 32, 1);
            size_t MemSize = WarpsPerBlock * sizeof(REAL);

            unsigned j = 0;
            while(lev_size > 0) {
                unsigned restCol = lev_size > TMPMEMNUM ? TMPMEMNUM : lev_size;
                dim3 dimGrid(restCol, 1);
                if (!PERTURB)
                    RL<<<dimGrid, dimBlock, MemSize>>>(sym_c_ptr_dev,
                                        sym_r_idx_dev,
                                        val_dev,
                                        l_col_ptr_dev,
                                        csr_r_ptr_dev,
                                        csr_c_idx_dev,
                                        csr_diag_ptr_dev,
                                        level_idx_dev,
                                        tmpMem,
                                        n,
                                        A_sym.level_ptr[i],
                                        j*TMPMEMNUM);
                else
                    RL_perturb<<<dimGrid, dimBlock, MemSize>>>(sym_c_ptr_dev,
                                        sym_r_idx_dev,
                                        val_dev,
                                        l_col_ptr_dev,
                                        csr_r_ptr_dev,
                                        csr_c_idx_dev,
                                        csr_diag_ptr_dev,
                                        level_idx_dev,
                                        tmpMem,
                                        n,
                                        A_sym.level_ptr[i],
                                        j*TMPMEMNUM,
                                        pert);
                j++;
                lev_size -= TMPMEMNUM;
            }
        }
        else if (lev_size > Nstreams) {
            //std::cout << "LEVEL " << i << " IS SMALL: " << lev_size << std::endl;
            dim3 dimBlock(1024, 1);
            size_t MemSize = 32 * sizeof(REAL);
            unsigned j = 0;
            while(lev_size > 0) {
                unsigned restCol = lev_size > TMPMEMNUM ? TMPMEMNUM : lev_size;
                dim3 dimGrid(restCol, 1);
                if (!PERTURB)
                    RL<<<dimGrid, dimBlock, MemSize>>>(sym_c_ptr_dev,
                                        sym_r_idx_dev,
                                        val_dev,
                                        l_col_ptr_dev,
                                        csr_r_ptr_dev,
                                        csr_c_idx_dev,
                                        csr_diag_ptr_dev,
                                        level_idx_dev,
                                        tmpMem,
                                        n,
                                        A_sym.level_ptr[i],
                                        j*TMPMEMNUM);
                else
                    RL_perturb<<<dimGrid, dimBlock, MemSize>>>(sym_c_ptr_dev,
                                        sym_r_idx_dev,
                                        val_dev,
                                        l_col_ptr_dev,
                                        csr_r_ptr_dev,
                                        csr_c_idx_dev,
                                        csr_diag_ptr_dev,
                                        level_idx_dev,
                                        tmpMem,
                                        n,
                                        A_sym.level_ptr[i],
                                        j*TMPMEMNUM,
                                        pert);
                j++;
                lev_size -= TMPMEMNUM;
            }
        }
        else { // "Big" levels
            //std::cout << "LEVEL " << i << " IS VERY SMALL: " << lev_size << std::endl;
            for (unsigned offset = 0; offset < lev_size; offset += Nstreams) {
                for (int j = 0; j < Nstreams; j++) {
                    if (j + offset < lev_size) {
                        const unsigned currentCol = A_sym.level_idx[A_sym.level_ptr[i] + j + offset];
                        const unsigned currentLColSize = A_sym.sym_c_ptr[currentCol + 1]
                            - A_sym.l_col_ptr[currentCol] - 1;
                        const unsigned subMatSize = A_sym.csr_r_ptr[currentCol + 1]
                            - A_sym.csr_diag_ptr[currentCol] - 1;

                        if (!PERTURB)
                            RL_onecol_factorizeCurrentCol<<<1, 1024, 0, streams[j]>>>(sym_c_ptr_dev,
                                                    sym_r_idx_dev,
                                                    val_dev,
                                                    l_col_ptr_dev,
                                                    currentCol,
                                                    tmpMem,
                                                    j,
                                                    n);
                        else
                            RL_onecol_factorizeCurrentCol_perturb<<<1, 1024, 0, streams[j]>>>(sym_c_ptr_dev,
                                                    sym_r_idx_dev,
                                                    val_dev,
                                                    l_col_ptr_dev,
                                                    currentCol,
                                                    tmpMem,
                                                    j,
                                                    n,
                                                    pert);
                        if (subMatSize > 0)
                            RL_onecol_updateSubmat<<<subMatSize, 1024, 0, streams[j]>>>(sym_c_ptr_dev,
                                                        sym_r_idx_dev,
                                                        val_dev,
                                                        csr_c_idx_dev,
                                                        csr_diag_ptr_dev,
                                                        currentCol,
                                                        tmpMem,
                                                        j,
                                                        n);
                        RL_onecol_cleartmpMem<<<1, 1024, 0, streams[j]>>>(sym_c_ptr_dev,
                                                    sym_r_idx_dev,
                                                    l_col_ptr_dev,
                                                    currentCol,
                                                    tmpMem,
                                                    j,
                                                    n);
                    }
                }
            }
        }
        cudaDeviceSynchronize();
    }

    //copy LU val back to main mem
    cudaMemcpy(&(A_sym.val[0]), val_dev, nnz * sizeof(REAL), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    out << "Total GPU time: " << time << " ms" << endl;

    cudaError_t cudaRet = cudaGetLastError();
    if (cudaRet != cudaSuccess) {
        out << cudaGetErrorName(cudaRet) << endl;
        out << cudaGetErrorString(cudaRet) << endl;
    }

#ifdef GLU_DEBUG
    //check NaN elements
    unsigned err_find = 0;
    for(unsigned i = 0; i < nnz; i++)
        if(isnan(A_sym.val[i]) || isinf(A_sym.val[i])) 
            err_find++;

    if (err_find != 0)
        err << "LU data check: " << " NaN found!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
#endif

    cudaFree(sym_c_ptr_dev);
    cudaFree(sym_r_idx_dev);
    cudaFree(val_dev);

    cudaFree(l_col_ptr_dev);
    cudaFree(csr_c_idx_dev);
    cudaFree(csr_r_ptr_dev);
    cudaFree(csr_diag_ptr_dev);

    cudaFree(level_idx_dev);
}
