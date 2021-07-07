/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/


#include <cuda.h>
#include <mpi.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char *argv[])
{
    int num_cuda_devices = 0;
    cudaGetDeviceCount(&num_cuda_devices);
    if (num_cuda_devices < 1) std::exit(-1);
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    assert(size > 1);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cudaSetDevice(rank);
    int *d_buf;
    int *buf;
    unsigned long len = 10;
    buf = (int *)malloc(sizeof(int) * len);
    for (int i = 0; i < len; ++i) {
        buf[i] = (i + 1) * (rank + 1);
    }
    cudaMalloc(&d_buf, sizeof(int) * len);
    cudaMemcpy(d_buf, buf, sizeof(int) * len, cudaMemcpyHostToDevice);
    if (rank == 0) {
        MPI_Send(d_buf, len, MPI_INT, 1, 12, MPI_COMM_WORLD);
    } else {
        MPI_Status status;
        MPI_Recv(d_buf, len, MPI_INT, 0, 12, MPI_COMM_WORLD, &status);
        for (int i = 0; i < len; ++i) {
            bool flag = (buf[i] == (i + 1) * 2);
            if (!flag) std::exit(-1);
        }
        cudaMemcpy(buf, d_buf, sizeof(int) * len, cudaMemcpyDeviceToHost);
        for (int i = 0; i < len; ++i) {
            bool flag = (buf[i] == (i + 1));
            if (!flag) std::exit(-1);
        }
    }
    cudaFree(d_buf);
    free(buf);
    MPI_Finalize();
    return 0;
}
