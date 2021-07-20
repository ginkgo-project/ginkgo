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


#include "core/solver/batch_direct_kernels.hpp"


#include "core/matrix/batch_struct.hpp"
#include "cuda/base/cublas_bindings.hpp"
#include "cuda/components/thread_ids.cuh"


namespace gko {
namespace kernels {
namespace cuda {
namespace batch_direct {


constexpr int default_block_size = 512;
#include "common/solver/batch_direct_kernels.hpp.inc"


namespace {

void check_batch(std::shared_ptr<const CudaExecutor> exec, const int nbatch,
                 const int *const info, const bool factorization)
{
    auto host_exec = exec->get_master();
    int *const h_info = host_exec->alloc<int>(nbatch);
    host_exec->copy_from(exec.get(), nbatch, info, h_info);
    for (int i = 0; i < nbatch; i++) {
        if (info[i] < 0 && factorization) {
            std::cerr << "Cublas batch factorization was given an invalid "
                      << "argument at the " << -1 * info[i] << "th position.\n";
        } else if (info[i] < 0 && !factorization) {
            std::cerr << "Cublas batch triangular solve was given an invalid "
                      << "argument at the " << -1 * info[i] << "th position.\n";
        } else if (info[i] > 0 && factorization) {
            std::cerr << "Cublas batch factorization: The " << info[i]
                      << "th matrix was singular.\n";
        }
    }
    host_exec->free(h_info);
}

}  // namespace


template <typename ValueType>
void apply(std::shared_ptr<const CudaExecutor> exec,
           matrix::BatchDense<ValueType> *const a,
           matrix::BatchDense<ValueType> *const b,
           gko::log::BatchLogData<ValueType> &logdata)
{
    const size_type num_batches = a->get_num_batch_entries();
    const int nbatch = static_cast<int>(num_batches);
    const int n = a->get_size().at()[0];
    const size_type stride = a->get_stride().at();
    const int lda = static_cast<int>(stride);
    const size_type b_stride = b->get_stride().at();
    const int nrhs = static_cast<int>(b->get_size().at()[1]);
    const int ldb = static_cast<int>(b_stride);

    int *const pivot_array = exec->alloc<int>(nbatch * n);
    int *const info_array = exec->alloc<int>(nbatch);
    ValueType **const matrices = exec->alloc<ValueType *>(nbatch);
    ValueType **const vectors = exec->alloc<ValueType *>(nbatch);
    const int nblk_1 = (nbatch - 1) / default_block_size + 1;
    setup_batch_pointers<<<nblk_1, default_block_size>>>(
        num_batches, n, stride, as_cuda_type(a->get_values()),
        as_cuda_type(matrices), b_stride, as_cuda_type(b->get_values()),
        as_cuda_type(vectors));

    auto handle = cublas::init();
    cublas::batch_getrf(handle, n, matrices, lda, pivot_array, info_array,
                        nbatch);
#ifdef DEBUG
    check_batch(exec, nbatch, info_array, true);
#endif
    cublas::batch_getrs(handle, CUBLAS_OP_N, n, nrhs, matrices, lda,
                        pivot_array, vectors, ldb, info_array, nbatch);
#ifdef DEBUG
    check_batch(exec, nbatch, info_array, false);
#endif
    cublas::destroy(handle);

    exec->free(matrices);
    exec->free(vectors);
    exec->free(pivot_array);
    exec->free(info_array);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DIRECT_APPLY_KERNEL);


template <typename ValueType>
void scale_and_copy(std::shared_ptr<const CudaExecutor> exec,
                    const matrix::BatchDense<ValueType> *const scaling_vec,
                    const matrix::BatchDense<ValueType> *const orig,
                    matrix::BatchDense<ValueType> *const scaled)
{
    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DIRECT_SCALE_AND_COPY);


}  // namespace batch_direct
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
