// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_FACTORIZATION_PAR_ILUT_SELECT_COMMON_CUH_
#define GKO_CUDA_FACTORIZATION_PAR_ILUT_SELECT_COMMON_CUH_


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
namespace cuda {
namespace par_ilut_factorization {


constexpr int default_block_size = 512;
constexpr int items_per_thread = 16;


template <typename ValueType, typename IndexType>
void sampleselect_count(std::shared_ptr<const DefaultExecutor> exec,
                        const ValueType* values, IndexType size,
                        remove_complex<ValueType>* tree, unsigned char* oracles,
                        IndexType* partial_counts, IndexType* total_counts);


template <typename IndexType>
struct sampleselect_bucket {
    IndexType idx;
    IndexType begin;
    IndexType size;
};


template <typename IndexType>
sampleselect_bucket<IndexType> sampleselect_find_bucket(
    std::shared_ptr<const DefaultExecutor> exec, IndexType* prefix_sum,
    IndexType rank);


}  // namespace par_ilut_factorization
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_FACTORIZATION_PAR_ILUT_SELECT_COMMON_CUH_
