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

#ifndef GKO_CORE_SOLVER_BATCH_GMRES_KERNELS_HPP_
#define GKO_CORE_SOLVER_BATCH_GMRES_KERNELS_HPP_


#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/stop/batch_stop_enum.hpp>


#include "core/log/batch_logging.hpp"

namespace gko {
namespace kernels {
namespace batch_gmres {


/**
 * Options controlling the batch Gmres solver.
 */
template <typename RealType>
struct BatchGmresOptions {
    int max_its;
    RealType residual_tol;
    int restart_num;
    ::gko::stop::batch::ToleranceType tol_type;
};


/**
 * Calculates the amount of in-solver storage needed by batch-Gmres.
 *
 * The calculation includes
 * multivectors (length of each vector: number of rows in system matrix) for
 * - r
 * - z
 * - w
 * - x
 * - helper
 * multivectors (length of each vector: restart ) for
 * - cs
 * - sn
 * - y
 * multivectors (length of each vector: restart + 1) for
 * - s
 * matrices for
 * - Hessenberg matrix
 * - Krylov subspace basis vectors
 * and small arrays for
 * - rhs_norms
 * - res_norms
 * - tmp_norms
 */
template <typename ValueType>
inline int local_memory_requirement(const int num_rows, const int num_rhs,
                                    const int restart_num)
{
    return (5 * num_rows * num_rhs + 3 * restart_num * num_rhs +
            (restart_num + 1) * num_rhs +
            restart_num * (restart_num + 1) * num_rhs +
            num_rows * (restart_num + 1) * num_rhs) *
               sizeof(ValueType) +
           3 * num_rhs * sizeof(typename gko::remove_complex<ValueType>);
}


struct StorageConfig {
    // rot storage
    bool rot_shared;

    // preconditioner storage
    bool prec_shared;

    // subspace storage
    bool subspace_shared;

    // hess storage
    bool hess_shared;

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

template <int align_bytes, typename value_type>
void set_gmem_stride_bytes(StorageConfig& sconf,
                           const int main_vecs_storage_bytes,
                           const int rot_vecs_storage_bytes,
                           const int prec_storage_bytes,
                           const int subspace_storage_bytes,
                           const int hess_storage_bytes)
{
    int gmem_stride = main_vecs_storage_bytes;
    gmem_stride += rot_vecs_storage_bytes;

    if (!sconf.prec_shared) {
        gmem_stride += prec_storage_bytes;
    }
    if (!sconf.subspace_shared) {
        gmem_stride += subspace_storage_bytes;
    }
    if (!sconf.hess_shared) {
        gmem_stride += hess_storage_bytes;
    }
    // align global memory chunks
    sconf.gmem_stride_bytes =
        gmem_stride > 0 ? ((gmem_stride - 1) / align_bytes + 1) * align_bytes
                        : 0;
}


}  // namespace


/**
 * Calculates the amount of in-solver storage needed by batch-gmres and
 * the split between shared and global memory.
 *
 * The calculation includes multivectors for
 * - r
 * - z
 * - w
 * - x
 * - helper
 * - cosine
 * - sine
 * - y
 * - s
 * - Hessenberg
 * - V (Krylov subspace vectors)
 * - precond
 * In addition, small arrays are needed for
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
 * @param restart The restart for GMRES
 * @param num_rhs  Number of right-hand-sides in the vectors.
 * @return  A struct containing allocation information specific to GMRES.
 */
template <typename Prectype, typename ValueType, int align_bytes = 32>
StorageConfig compute_shared_storage(const int shared_mem_per_blk,
                                     const int num_rows, const int num_nz,
                                     const int num_rhs, const int restart)
{
    using real_type = remove_complex<ValueType>;
    const int mvecs_size_byte = num_rows * num_rhs * sizeof(ValueType);
    // To simplify, all rotation vectors sizes are 'rounded up' to restart+1
    const int rvecs_size_byte = (restart + 1) * sizeof(ValueType);
    const int num_total_vecs = 9;
    int num_rmain_vecs = 5;
    int num_rrot_vecs = 4;
    const int prec_storage =
        Prectype::dynamic_work_size(num_rows, num_nz) * sizeof(ValueType);

    const int hess_storage = restart * (restart + 1) * sizeof(ValueType);
    const int subspace_storage = num_rows * (restart + 1) * sizeof(ValueType);

    int rem_shared = shared_mem_per_blk;
    StorageConfig sconf{false, false,          false, false,
                        0,     num_total_vecs, 0,     num_rows};

    if (rem_shared <= 0) {
        set_gmem_stride_bytes<align_bytes, ValueType>(
            sconf, num_rmain_vecs * mvecs_size_byte,
            num_rrot_vecs * mvecs_size_byte, prec_storage, subspace_storage,
            hess_storage);
        return sconf;
    }
    // Main vecs
    int num_shared_main_vecs = rem_shared / mvecs_size_byte;
    num_shared_main_vecs = min(num_shared_main_vecs, num_rmain_vecs);
    sconf.n_shared += num_shared_main_vecs;
    num_rmain_vecs -= num_shared_main_vecs;
    if (num_rmain_vecs > 0) {
        sconf.n_global -= sconf.n_shared;
        set_gmem_stride_bytes<align_bytes, ValueType>(
            sconf, num_rmain_vecs * mvecs_size_byte,
            num_rrot_vecs * rvecs_size_byte, prec_storage, subspace_storage,
            hess_storage);
        return sconf;
    }
    rem_shared -= num_shared_main_vecs * mvecs_size_byte;
    // rotation vecs
    int num_shared_rot_vecs = rem_shared / rvecs_size_byte;
    num_shared_rot_vecs = min(num_shared_rot_vecs, num_rrot_vecs);
    sconf.n_shared += num_shared_rot_vecs;
    num_rrot_vecs -= num_shared_rot_vecs;
    if (num_rrot_vecs > 0) {
        sconf.n_global -= sconf.n_shared;
        set_gmem_stride_bytes<align_bytes, ValueType>(
            sconf, num_rmain_vecs * mvecs_size_byte,
            num_rrot_vecs * rvecs_size_byte, prec_storage, subspace_storage,
            hess_storage);
        return sconf;
    }
    // Other
    rem_shared -= num_shared_rot_vecs * rvecs_size_byte;
    if (rem_shared >= hess_storage) {
        sconf.hess_shared = true;
        rem_shared -= hess_storage;
    }
    if (rem_shared >= subspace_storage) {
        sconf.subspace_shared = true;
        rem_shared -= subspace_storage;
    }
    if (rem_shared >= prec_storage) {
        sconf.prec_shared = true;
        rem_shared -= prec_storage;
    }

    sconf.n_global -= sconf.n_shared;
    set_gmem_stride_bytes<align_bytes, ValueType>(
        sconf, num_rmain_vecs * mvecs_size_byte,
        num_rrot_vecs * rvecs_size_byte, prec_storage, subspace_storage,
        hess_storage);
    return sconf;
}


#define GKO_DECLARE_BATCH_GMRES_APPLY_KERNEL(_type)                \
    void apply(std::shared_ptr<const DefaultExecutor> exec,        \
               const gko::kernels::batch_gmres::BatchGmresOptions< \
                   remove_complex<_type>>& options,                \
               const BatchLinOp* a, const BatchLinOp* precon,      \
               const matrix::BatchDense<_type>* const b,           \
               matrix::BatchDense<_type>* const x,                 \
               gko::log::BatchLogData<_type>& logdata)


#define GKO_DECLARE_ALL_AS_TEMPLATES \
    template <typename ValueType>    \
    GKO_DECLARE_BATCH_GMRES_APPLY_KERNEL(ValueType)


}  // namespace batch_gmres


namespace omp {
namespace batch_gmres {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_gmres
}  // namespace omp


namespace cuda {
namespace batch_gmres {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_gmres
}  // namespace cuda


namespace reference {
namespace batch_gmres {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_gmres
}  // namespace reference


namespace hip {
namespace batch_gmres {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_gmres
}  // namespace hip


namespace dpcpp {
namespace batch_gmres {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_gmres
}  // namespace dpcpp


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_BATCH_GMRES_KERNELS_HPP_
