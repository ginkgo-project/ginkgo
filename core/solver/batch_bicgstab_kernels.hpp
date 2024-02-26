// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_BATCH_BICGSTAB_KERNELS_HPP_
#define GKO_CORE_SOLVER_BATCH_BICGSTAB_KERNELS_HPP_


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/log/batch_logger.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/batch_ell.hpp>
#include <ginkgo/core/stop/batch_stop_enum.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace batch_bicgstab {


/**
 * Options controlling the batch Bicgstab solver.
 */
template <typename RealType>
struct settings {
    static_assert(std::is_same<RealType, remove_complex<RealType>>::value,
                  "Template parameter must be a real type");
    int max_iterations;
    RealType residual_tol;
    ::gko::batch::stop::tolerance_type tol_type;
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
}


struct storage_config {
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


template <int align_bytes>
void set_gmem_stride_bytes(storage_config& sconf,
                           const int multi_vector_size_bytes,
                           const int prec_storage_bytes)
{
    int gmem_stride = sconf.n_global * multi_vector_size_bytes;
    if (!sconf.prec_shared) {
        gmem_stride += prec_storage_bytes;
    }
    // align global memory chunks
    sconf.gmem_stride_bytes =
        gmem_stride > 0 ? ceildiv(gmem_stride, align_bytes) * align_bytes : 0;
}


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
 *
 * @param available_shared_mem  The amount of shared memory per block to use
 * for keeping intermediate vectors. In case keeping the matrix in L1 cache etc.
 *   should be prioritized, the cache configuration must be updated separately
 *   and the needed space should be subtracted before passing to this
 *   function.
 * @param num_rows  Size of the matrix.
 * @param num_nz  Number of nonzeros in the matrix
 * @param num_rhs  Number of right-hand-sides in the vectors.
 * @return  A struct containing allocation information specific to Bicgstab.
 */
template <typename Prectype, typename ValueType, int align_bytes = 32>
storage_config compute_shared_storage(const int available_shared_mem,
                                      const int num_rows, const int num_nz,
                                      const int num_rhs)
{
    using real_type = remove_complex<ValueType>;
    const int vec_size = num_rows * num_rhs * sizeof(ValueType);
    const int num_main_vecs = 9;
    const int prec_storage =
        Prectype::dynamic_work_size(num_rows, num_nz) * sizeof(ValueType);
    int rem_shared = available_shared_mem;
    // Set default values. Initially all vecs are in global memory.
    // {prec_shared, n_shared, n_global, gmem_stride_bytes, padded_vec_len}
    storage_config sconf{false, 0, num_main_vecs, 0, num_rows};
    // If available shared mem is zero, set all vecs to global.
    if (rem_shared <= 0) {
        set_gmem_stride_bytes<align_bytes>(sconf, vec_size, prec_storage);
        return sconf;
    }
    // Compute the number of vecs that can be stored in shared memory and assign
    // the rest to global memory.
    const int initial_vecs_available = rem_shared / vec_size;
    const int num_vecs_shared = min(initial_vecs_available, num_main_vecs);
    sconf.n_shared += num_vecs_shared;
    sconf.n_global -= num_vecs_shared;
    rem_shared -= num_vecs_shared * vec_size;
    // Set the storage configuration with preconditioner workspace in global if
    // there are any vectors in global memory.
    if (sconf.n_global > 0) {
        set_gmem_stride_bytes<align_bytes>(sconf, vec_size, prec_storage);
        return sconf;
    }
    // If more shared memory space is available and preconditioner workspace is
    // needed, enable preconditioner workspace to use shared memory.
    if (rem_shared >= prec_storage && prec_storage > 0) {
        sconf.prec_shared = true;
        rem_shared -= prec_storage;
    }
    // Set the global storage config and align to align_bytes bytes.
    set_gmem_stride_bytes<align_bytes>(sconf, vec_size, prec_storage);
    return sconf;
}


}  // namespace batch_bicgstab


#define GKO_DECLARE_BATCH_BICGSTAB_APPLY_KERNEL(_type)                       \
    void apply(                                                              \
        std::shared_ptr<const DefaultExecutor> exec,                         \
        const gko::kernels::batch_bicgstab::settings<remove_complex<_type>>& \
            options,                                                         \
        const batch::BatchLinOp* a, const batch::BatchLinOp* preconditioner, \
        const batch::MultiVector<_type>* b, batch::MultiVector<_type>* x,    \
        gko::batch::log::detail::log_data<remove_complex<_type>>& logdata)


#define GKO_DECLARE_ALL_AS_TEMPLATES \
    template <typename ValueType>    \
    GKO_DECLARE_BATCH_BICGSTAB_APPLY_KERNEL(ValueType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(batch_bicgstab,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_BATCH_BICGSTAB_KERNELS_HPP_
