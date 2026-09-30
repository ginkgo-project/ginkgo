// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MULTIGRID_PMIS_KERNELS_HPP_
#define GKO_CORE_MULTIGRID_PMIS_KERNELS_HPP_


#include <memory>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace pmis {


constexpr int coarse = 1;
constexpr int fine = 0;
constexpr int unassigned = -1;
// Use a fixed seed such that the coarsening is reproducible from run to run
// and consistent among the backends.
constexpr uint64 random_seed = 42;


// Stafford Mix13, the finalizer of splitmix64 (Steele, Lea & Flood, OOPSLA
// 2014, https://dl.acm.org/doi/10.1145/2714064.2660195; reference
// implementation at https://prng.di.unimi.it/splitmix64.c). It is used as a
// hash of the node's global index, so that a node draws the same value
// independently of the rank that owns it.
GKO_ATTRIBUTES GKO_INLINE float random_weight_from_index(uint64 idx)
{
    auto z = idx + random_seed;
    z = (z ^ (z >> 30)) * uint64{0xbf58476d1ce4e5b9};
    z = (z ^ (z >> 27)) * uint64{0x94d049bb133111eb};
    z = z ^ (z >> 31);
    // the top 24 bits are exact in a float significand and never round up to 1
    return static_cast<float>(z >> 40) * (1.0f / 16777216.0f);
}


// Per-row workspace shared by the two interpolation passes: the accumulate
// pass fills every member except cursor, the emit pass reads them and advances
// cursor. The members are raw pointers so that the caller can map them to
// device types and pass the workspace by value. The enable flags are int
// because array::fill has no bool instantiation.
template <typename ValueType, typename IndexType>
struct interpolation_workspace {
    ValueType* pos;
    ValueType* pos_divisor;
    ValueType* neg;
    ValueType* neg_divisor;
    ValueType* diag;
    int* enable_pos;
    int* enable_neg;
    IndexType* cursor;
};


// These kernels operate on one block at a time: a distributed matrix calls
// them once per block, a local matrix once with has_diagonal = true.
// compute_row_maxabs accumulates into row_maxabs, which the caller has to zero
// beforehand.
#define GKO_DECLARE_PMIS_COMPUTE_ROW_MAXABS_KERNEL(ValueType, IndexType)  \
    void compute_row_maxabs(std::shared_ptr<const DefaultExecutor> exec,  \
                            const matrix::Csr<ValueType, IndexType>* csr, \
                            bool has_diagonal,                            \
                            remove_complex<ValueType>* row_maxabs)


#define GKO_DECLARE_PMIS_COMPUTE_STRONG_DEP_ROW_KERNEL(ValueType, IndexType)  \
    void compute_strong_dep_row(std::shared_ptr<const DefaultExecutor> exec,  \
                                const matrix::Csr<ValueType, IndexType>* csr, \
                                bool has_diagonal,                            \
                                const remove_complex<ValueType>* row_maxabs,  \
                                remove_complex<ValueType> strength_threshold, \
                                IndexType* sparsity_rows)

#define GKO_DECLARE_PMIS_COMPUTE_STRONG_DEP_KERNEL(ValueType, IndexType) \
    void compute_strong_dep(                                             \
        std::shared_ptr<const DefaultExecutor> exec,                     \
        const matrix::Csr<ValueType, IndexType>* csr, bool has_diagonal, \
        const remove_complex<ValueType>* row_maxabs,                     \
        remove_complex<ValueType> strength_threshold,                    \
        matrix::SparsityCsr<ValueType, IndexType>* strong_dep)

// The weight of a node is its in-degree in the strength graph plus a draw in
// [0, 1). A node on which nothing strongly depends is marked fine immediately.
// counts is an input because the distributed path builds it by accumulating
// the halo contributions back to their owners.
#define GKO_DECLARE_PMIS_INITIALIZE_WEIGHT_AND_STATUS_KERNEL(            \
    ValueType, LocalIndexType, GlobalIndexType)                          \
    void initialize_weight_and_status(                                   \
        std::shared_ptr<const DefaultExecutor> exec, size_type num,      \
        const LocalIndexType* counts, const GlobalIndexType* global_idx, \
        ValueType* weight, int* status)

// Scatter-add out[idxs[i]] += values[i], or += 1 if values is null, which lets
// the column counts avoid an nnz-sized array of ones. The indices may repeat,
// so the update has to be atomic. This is why the kernel is backend-specific:
// the unified kernels provide no atomics. Using IndexType rather than
// size_type also allows a 32-bit index to use the native atomicAdd of cuda/hip
// instead of a compare-and-swap loop.
#define GKO_DECLARE_PMIS_ADD_AT_INDICES(IndexType)                   \
    void add_at_indices(std::shared_ptr<const DefaultExecutor> exec, \
                        size_type num, const IndexType* idxs,        \
                        const IndexType* values, IndexType* out)

// The selection is split into two phases because a distributed matrix has to
// exchange new_status between them. select only downgrades coarse to
// unassigned and mark_fine only upgrades unassigned to fine, so the per-block
// calls commute. The tie-break requires a total order across ranks, hence the
// global index.
#define GKO_DECLARE_PMIS_CLASSIFY_SELECT_KERNEL(ValueType, LocalIndexType, \
                                                GlobalIndexType)           \
    void classify_select(                                                  \
        std::shared_ptr<const DefaultExecutor> exec,                       \
        const remove_complex<ValueType>* weight,                           \
        const GlobalIndexType* global_idx, LocalIndexType col_offset,      \
        const matrix::SparsityCsr<ValueType, LocalIndexType>* strong_dep,  \
        const int* status, int* new_status)

#define GKO_DECLARE_PMIS_CLASSIFY_MARK_FINE_KERNEL(ValueType, IndexType)   \
    void classify_mark_fine(                                               \
        std::shared_ptr<const DefaultExecutor> exec, IndexType col_offset, \
        const matrix::SparsityCsr<ValueType, IndexType>* strong_dep,       \
        int* new_status)

#define GKO_DECLARE_COUNT_KERNEL                                           \
    void count(std::shared_ptr<const DefaultExecutor> exec, size_type num, \
               const int* status, size_type* num_unassigned)

// Opens a selection round: every unassigned node tentatively becomes coarse,
// which select then downgrades block by block.
#define GKO_DECLARE_PMIS_CLASSIFY_SEED_KERNEL                       \
    void classify_seed(std::shared_ptr<const DefaultExecutor> exec, \
                       size_type num, const int* status, int* new_status)

// Accumulates into prolong_row_count and skips coarse rows, whose single
// identity entry is seeded by the caller. The caller has to set
// prolong_row_count to 1 for coarse rows and 0 otherwise before the first
// call.
#define GKO_DECLARE_DIRECT_INTERPOLATION_ROW_COUNT(ValueType, IndexType)   \
    void direct_interpolation_row_count(                                   \
        std::shared_ptr<const DefaultExecutor> exec, IndexType col_offset, \
        const matrix::SparsityCsr<ValueType, IndexType>* strong_dep,       \
        const int* status, IndexType* prolong_row_count)

// Coarse global column of a local node: a C-point gets the partition offset
// plus its local coarse index, an F-point gets an invalid_index sentinel. The
// local path has no halo and gathers from the prefix-summed coarse_map
// instead.
#define GKO_DECLARE_PMIS_COARSE_GLOBAL_INDEX(LocalIndexType, GlobalIndexType) \
    void coarse_global_index(                                                 \
        std::shared_ptr<const DefaultExecutor> exec, size_type num,           \
        GlobalIndexType offset, const int* status,                            \
        const LocalIndexType* coarse_map, GlobalIndexType* coarse_global)

// Direct interpolation runs in three passes, because alpha and beta sum over
// the whole row while the emission divides by them: one pass for the identity
// entries of the coarse rows, one accumulating over every block, and one
// emitting. All three are driven by the caller, see multigrid/pmis.cpp. They
// determine whether a column is coarse from status rather than from
// coarse_map, which cannot span the halo, and emit node indices that the
// caller maps with a gather.
#define GKO_DECLARE_DIRECT_INTERPOLATION_ACCUMULATE(ValueType, IndexType)      \
    void direct_interpolation_accumulate(                                      \
        std::shared_ptr<const DefaultExecutor> exec,                           \
        const matrix::Csr<ValueType, IndexType>* csr, bool has_diagonal,       \
        IndexType col_offset, const remove_complex<ValueType>* row_maxabs,     \
        const remove_complex<ValueType> strength_threshold, const int* status, \
        kernels::pmis::interpolation_workspace<ValueType, IndexType> ws)

#define GKO_DECLARE_DIRECT_INTERPOLATION_EMIT(ValueType, IndexType)            \
    void direct_interpolation_emit(                                            \
        std::shared_ptr<const DefaultExecutor> exec,                           \
        const matrix::Csr<ValueType, IndexType>* csr, bool has_diagonal,       \
        IndexType col_offset, const remove_complex<ValueType>* row_maxabs,     \
        const remove_complex<ValueType> strength_threshold, const int* status, \
        kernels::pmis::interpolation_workspace<ValueType, IndexType> ws,       \
        IndexType* prolong_col_idxs, ValueType* prolong_values)

#define GKO_DECLARE_DIRECT_INTERPOLATION_FILL_COARSE_ROWS(ValueType,     \
                                                          IndexType)     \
    void direct_interpolation_fill_coarse_rows(                          \
        std::shared_ptr<const DefaultExecutor> exec, size_type num_rows, \
        const int* status, const IndexType* prolong_row_ptrs,            \
        IndexType* prolong_col_idxs, ValueType* prolong_values)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                       \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_PMIS_COMPUTE_ROW_MAXABS_KERNEL(ValueType, IndexType);      \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_PMIS_COMPUTE_STRONG_DEP_ROW_KERNEL(ValueType, IndexType);  \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_PMIS_COMPUTE_STRONG_DEP_KERNEL(ValueType, IndexType);      \
    template <typename ValueType, typename LocalIndexType,                 \
              typename GlobalIndexType>                                    \
    GKO_DECLARE_PMIS_INITIALIZE_WEIGHT_AND_STATUS_KERNEL(                  \
        ValueType, LocalIndexType, GlobalIndexType);                       \
    template <typename IndexType>                                          \
    GKO_DECLARE_PMIS_ADD_AT_INDICES(IndexType);                            \
    template <typename ValueType, typename LocalIndexType,                 \
              typename GlobalIndexType>                                    \
    GKO_DECLARE_PMIS_CLASSIFY_SELECT_KERNEL(ValueType, LocalIndexType,     \
                                            GlobalIndexType);              \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_PMIS_CLASSIFY_MARK_FINE_KERNEL(ValueType, IndexType);      \
    GKO_DECLARE_COUNT_KERNEL;                                              \
    GKO_DECLARE_PMIS_CLASSIFY_SEED_KERNEL;                                 \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_DIRECT_INTERPOLATION_ROW_COUNT(ValueType, IndexType);      \
    template <typename LocalIndexType, typename GlobalIndexType>           \
    GKO_DECLARE_PMIS_COARSE_GLOBAL_INDEX(LocalIndexType, GlobalIndexType); \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_DIRECT_INTERPOLATION_ACCUMULATE(ValueType, IndexType);     \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_DIRECT_INTERPOLATION_EMIT(ValueType, IndexType);           \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_DIRECT_INTERPOLATION_FILL_COARSE_ROWS(ValueType, IndexType)


}  // namespace pmis


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(pmis, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MULTIGRID_PMIS_KERNELS_HPP_
