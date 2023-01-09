/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#ifndef GKO_CORE_SOLVER_BATCH_CG_KERNELS_HPP_
#define GKO_CORE_SOLVER_BATCH_CG_KERNELS_HPP_


#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/stop/batch_stop_enum.hpp>


#include "core/log/batch_logging.hpp"

namespace gko {
namespace kernels {
namespace batch_cg {


/**
 * Options controlling the batch Cg solver.
 */
template <typename RealType>
struct BatchCgOptions {
    int max_its;
    RealType residual_tol;
    ::gko::stop::batch::ToleranceType tol_type;
};


/**
 * Calculates the amount of in-solver storage needed by batch-Cg.
 *
 * The calculation includes multivectors for
 * - r
 * - z
 * - p
 * - Ap
 * - x
 * and small arrays for
 * - rho_old
 * - rho_new
 * - alpha
 * - rhs_norms
 * - res_norms
 */
template <typename ValueType>
inline int local_memory_requirement(const int num_rows, const int num_rhs)
{
    return (5 * num_rows * num_rhs + 3 * num_rhs) * sizeof(ValueType) +
           2 * num_rhs * sizeof(typename gko::remove_complex<ValueType>);
}


/**
 * Encodes information about where solver vectors are stored, as well as
 * hardware-specific problem dimensions.
 *
 * @note Vector counts, such as `n_shared` and `n_global`, do not include
 *       preconditioner storage.
 */
struct StorageConfig {
    // preconditioner storage
    bool prec_shared;

    // total number of shared vectors
    int n_shared;

    // number of vectors in global memory
    int n_global;

    // global stride from one batch entry to the next
    int gmem_stride_bytes;

    // padded vector length
    int padded_vec_len;
};


namespace {

template <int align_bytes>
void set_gmem_stride_bytes(StorageConfig& sconf,
                           const int multi_vector_size_bytes,
                           const int prec_storage_bytes)
{
    int gmem_stride = sconf.n_global * multi_vector_size_bytes;
    if (!sconf.prec_shared) {
        gmem_stride += prec_storage_bytes;
    }
    // align global memory chunks
    sconf.gmem_stride_bytes =
        gmem_stride > 0 ? ((gmem_stride - 1) / align_bytes + 1) * align_bytes
                        : 0;
}

}  // namespace


/**
 * Calculates the amount of in-solver storage needed by batch-Cg and
 * the split between shared and global memory.
 *
 * The calculation includes multivectors for
 * - r
 * - p
 * - x
 * In addition, small arrays are needed for
 * - rho_old
 * - rho_new
 * - alpha
 * - rhs_norms
 * - res_norms
 *
 * @param shared_mem_per_blk  The amount of shared memory per block to use for
 *   keeping intermediate vectors. In case keeping the matrix in L1 cache etc.
 *   should be prioritized, the cache configuration must be updated separately
 *   and the needed space should be subtracted before passing to this
 *   function.
 * @param num_rows  Size of the matrix.
 * @param num_nz  Number of nonzeros in the matrix
 * @param num_rhs  Number of right-hand-sides in the vectors.
 * @return  A struct containing allocation information specific to Cg.
 */
template <typename Prectype, typename ValueType, int align_bytes = 32>
StorageConfig compute_shared_storage(const int shared_mem_per_blk,
                                     const int num_rows, const int num_nz,
                                     const int num_rhs)
{
    using real_type = remove_complex<ValueType>;
    const int vec_size = num_rows * num_rhs * sizeof(ValueType);
    const int num_priority_vecs = 4;
    const int prec_storage =
        Prectype::dynamic_work_size(num_rows, num_nz) * sizeof(ValueType);
    int rem_shared = shared_mem_per_blk;
    const int num_cg_vecs{6};
    StorageConfig sconf{false, 0, num_cg_vecs, 0, num_rows};
    if (rem_shared <= 0) {
        set_gmem_stride_bytes<align_bytes>(sconf, vec_size, prec_storage);
        return sconf;
    }
    const int initial_vecs_available = rem_shared / vec_size;
    const int priority_available = initial_vecs_available >= num_priority_vecs
                                       ? num_priority_vecs
                                       : initial_vecs_available;
    sconf.n_shared += priority_available;
    sconf.n_global -= priority_available;
    // for simplicity, we don't allocate anything else in shared
    //  if all the spmv vectors were not.
    if (priority_available < num_priority_vecs) {
        set_gmem_stride_bytes<align_bytes>(sconf, vec_size, prec_storage);
        return sconf;
    }
    rem_shared -= priority_available * vec_size;
    if (rem_shared >= prec_storage) {
        sconf.prec_shared = true;
        rem_shared -= prec_storage;
    }
    const int shared_other_vecs =
        rem_shared / vec_size >= 0 ? rem_shared / vec_size : 0;
    sconf.n_shared += shared_other_vecs;
    sconf.n_shared = min(sconf.n_shared, num_cg_vecs);
    sconf.n_global -= shared_other_vecs;
    sconf.n_global = max(sconf.n_global, 0);
    set_gmem_stride_bytes<align_bytes>(sconf, vec_size, prec_storage);
    return sconf;
}


#define GKO_DECLARE_BATCH_CG_APPLY_KERNEL(_type)                             \
    void apply(                                                              \
        std::shared_ptr<const DefaultExecutor> exec,                         \
        const gko::kernels::batch_cg::BatchCgOptions<remove_complex<_type>>& \
            options,                                                         \
        const BatchLinOp* const a, const BatchLinOp* const precon,           \
        const matrix::BatchDense<_type>* const b,                            \
        matrix::BatchDense<_type>* const x,                                  \
        gko::log::BatchLogData<_type>& logdata)


#define GKO_DECLARE_ALL_AS_TEMPLATES \
    template <typename ValueType>    \
    GKO_DECLARE_BATCH_CG_APPLY_KERNEL(ValueType)


}  // namespace batch_cg


namespace omp {
namespace batch_cg {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_cg
}  // namespace omp


namespace cuda {
namespace batch_cg {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_cg
}  // namespace cuda


namespace reference {
namespace batch_cg {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_cg
}  // namespace reference


namespace hip {
namespace batch_cg {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_cg
}  // namespace hip


namespace dpcpp {
namespace batch_cg {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_cg
}  // namespace dpcpp


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_BATCH_CG_KERNELS_HPP_
