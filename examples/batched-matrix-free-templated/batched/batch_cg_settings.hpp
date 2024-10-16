// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/core/stop/batch_stop_enum.hpp>


namespace gko {
namespace kernels {

// TODO: update when splitting compilation
constexpr bool cg_no_shared_vecs = true;


namespace batch_cg {


template <typename RealType>
struct settings {
    static_assert(std::is_same<RealType, remove_complex<RealType>>::value,
                  "Template parameter must be a real type");
    int max_iterations;
    RealType residual_tol;
    batch::stop::tolerance_type tol_type;
};

template <typename ValueType>
inline int local_memory_requirement(const int num_rows, const int num_rhs)
{
    return (5 * num_rows * num_rhs + 3 * num_rhs) * sizeof(ValueType) +
           2 * num_rhs * sizeof(typename gko::remove_complex<ValueType>);
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
    sconf.gmem_stride_bytes = ceildiv(gmem_stride, align_bytes) * align_bytes;
}


template <typename Prectype, typename ValueType, int align_bytes = 32>
storage_config compute_shared_storage(const int available_shared_mem,
                                      const int num_rows, const int num_nz,
                                      const int num_rhs)
{
    using real_type = remove_complex<ValueType>;
    const int vec_bytes = num_rows * num_rhs * sizeof(ValueType);
    const int num_main_vecs = 5;
    const int prec_storage = Prectype::dynamic_work_size(num_rows, num_nz);
    int rem_shared = available_shared_mem;
    // Set default values. Initially all vecs are in global memory.
    // {prec_shared, n_shared, n_global, gmem_stride_bytes, padded_vec_len}
    storage_config sconf{false, 0, num_main_vecs, 0, num_rows};
    // If available shared mem is zero, set all vecs to global.
    set_gmem_stride_bytes<align_bytes>(sconf, vec_bytes, prec_storage);
    return sconf;
}


}  // namespace batch_cg
}  // namespace kernels
}  // namespace gko
