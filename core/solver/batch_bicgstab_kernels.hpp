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

#ifndef GKO_CORE_SOLVER_BATCH_BICGSTAB_KERNELS_HPP_
#define GKO_CORE_SOLVER_BATCH_BICGSTAB_KERNELS_HPP_


#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/preconditioner/batch_preconditioner_types.hpp>
#include <ginkgo/core/stop/batch_stop_enum.hpp>


#include "core/log/batch_logging.hpp"

namespace gko {
namespace kernels {
namespace batch_bicgstab {


/**
 * Options controlling the batch Bicgstab solver.
 */
template <typename RealType>
struct BatchBicgstabOptions {
    preconditioner::batch::type preconditioner;
    int max_its;
    RealType residual_tol;
    ::gko::stop::batch::ToleranceType tol_type;
    int num_sh_vecs = 10;
};


/**
 * Calculates the amount of in-solver storage needed by batch-Bicgstab.
 *
 * The calculation includes multivectors for
 * - r
 * - r_hat
 * - p
 * - p_hat
 * - v
 * - s
 * - s_hat
 * - t
 * - x
 * Note: small arrays for
 * - rho_old
 * - rho_new
 * - omega
 * - alpha
 * - temp
 * - rhs_norms
 * - res_norms
 * are currently not accounted for as they are in static shared memory.
 */
template <typename ValueType>
inline int local_memory_requirement(const int num_rows, const int num_rhs)
{
    return (9 * num_rows * num_rhs) * sizeof(ValueType);
    //+ 2 * num_rhs * sizeof(typename gko::remove_complex<ValueType>);
}


struct StorageConfig {
    // preconditioner storage
    bool prec_shared;

    // total number of shared vectors
    int n_shared;

    // number of vectors in global memory
    int n_global;

    // global stride from one batch entry to the next
    int gmem_stride_bytes;
};


namespace {

template <typename ValueType, int align_bytes>
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
 * Calculates the amount of in-solver storage needed by batch-Bicgstab and
 * the split between shared and global memory.
 *
 * The calculation includes multivectors for
 * - r
 * - r_hat
 * - p
 * - p_hat
 * - v
 * - s
 * - s_hat
 * - t
 * - x
 * In addition, small arrays are needed for
 * - rho_old
 * - rho_new
 * - omega
 * - alpha
 * - temp
 * - rhs_norms
 * - res_norms
 */
template <typename Prectype, typename ValueType, int align_bytes = 32>
StorageConfig compute_shared_storage(const int shared_mem_per_sm,
                                     const int num_rows, const int num_nz,
                                     const int num_rhs)
{
    using real_type = remove_complex<ValueType>;
    const int vec_size = num_rows * num_rhs * sizeof(ValueType);
    // const int padded_vec_size = ((vec_size - 1) / align_bytes + 1) *
    // align_bytes; const int padded_multivec_len = padded_vec_size /
    // sizeof(ValueType); assert(padded_multivec_len % sizeof(ValueType) == 0);
    const int num_value_scalars = 5 * num_rhs;
    const int num_real_scalars = 2 * num_rhs;
    const int num_priority_vecs = 4;
    const int prec_storage =
        Prectype::dynamic_work_size(num_rows, num_nz) * sizeof(ValueType);
    // Prioritize caching of matrix in L1 cache
    // for now, this is hard-coded for CSR
    // TODO: add functions to batch matrix formats to return storage per batch
    // const int matrix_storage = (num_rows + 1) * sizeof(int) +
    //                           num_nz * (sizeof(int) + sizeof(ValueType));
    int rem_shared = shared_mem_per_sm - /*matrix_storage -*/
                     num_value_scalars * sizeof(ValueType) -
                     num_real_scalars * sizeof(real_type);
    StorageConfig sconf{false, 0, 9, 0};
    if (rem_shared <= 0) {
        set_gmem_stride_bytes<ValueType, align_bytes>(sconf, vec_size,
                                                      prec_storage);
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
        set_gmem_stride_bytes<ValueType, align_bytes>(sconf, vec_size,
                                                      prec_storage);
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
    if (sconf.n_shared > 9) {
        sconf.n_shared = 9;
    }
    sconf.n_global -= shared_other_vecs;
    if (sconf.n_global < 0) {
        sconf.n_global = 0;
    }
    set_gmem_stride_bytes<ValueType, align_bytes>(sconf, vec_size,
                                                  prec_storage);
    return sconf;
}


#define GKO_DECLARE_BATCH_BICGSTAB_APPLY_KERNEL(_type)                   \
    void apply(std::shared_ptr<const DefaultExecutor> exec,              \
               const gko::kernels::batch_bicgstab::BatchBicgstabOptions< \
                   remove_complex<_type>>& options,                      \
               const BatchLinOp* const a,                                \
               const matrix::BatchDense<_type>* const b,                 \
               matrix::BatchDense<_type>* const x,                       \
               gko::log::BatchLogData<_type>& logdata)


#define GKO_DECLARE_ALL_AS_TEMPLATES \
    template <typename ValueType>    \
    GKO_DECLARE_BATCH_BICGSTAB_APPLY_KERNEL(ValueType)


}  // namespace batch_bicgstab


namespace omp {
namespace batch_bicgstab {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_bicgstab
}  // namespace omp


namespace cuda {
namespace batch_bicgstab {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_bicgstab
}  // namespace cuda


namespace reference {
namespace batch_bicgstab {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_bicgstab
}  // namespace reference


namespace hip {
namespace batch_bicgstab {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_bicgstab
}  // namespace hip


namespace dpcpp {
namespace batch_bicgstab {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_bicgstab
}  // namespace dpcpp


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_BATCH_BICGSTAB_KERNELS_HPP_
