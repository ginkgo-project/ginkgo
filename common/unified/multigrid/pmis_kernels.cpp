// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/pmis_kernels.hpp"

#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include "common/unified/base/kernel_launch.hpp"
#include "common/unified/base/kernel_launch_reduction.hpp"
#include "core/base/array_access.hpp"
#include "core/components/prefix_sum_kernels.hpp"

#include <random>


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The Pmis namespace.
 *
 * @ingroup pmis
 */
namespace pmis {


template <typename ValueType, typename IndexType>
void compute_strong_dep_row(std::shared_ptr<const DefaultExecutor> exec,
                            const matrix::Csr<ValueType, IndexType>* csr,
                            remove_complex<ValueType> strength_threshold,
                            IndexType* sparsity_rows)
{
    std::vector<IndexType> max_values(csr->get_size()[1]);

    for (IndexType col = 0; col < csr->get_size()[1]; ++col) {
        remove_complex<ValueType> max_val = 0;
        for (IndexType row = 0; row < csr->get_size()[0]; ++row) {
            for (auto idx = csr->get_const_row_ptrs()[row]; idx < csr->get_const_row_ptrs()[row + 1]; ++idx) {
                if (csr->get_const_col_idxs()[idx] == col && row != col) {
                    max_val = std::max(max_val, gko::abs(csr->get_const_values()[idx]));
                }
            }
        }
        max_values[col] = max_val;
    }

    for (IndexType col = 0; col < csr->get_size()[1]; ++col) {
        for (IndexType row = 0; row < csr->get_size()[0]; ++row) {
            sparsity_rows[row] = 0;
            for (auto idx = csr->get_const_row_ptrs()[row]; idx < csr->get_const_row_ptrs()[row + 1]; ++idx) {
                if (gko::abs(csr->get_const_values()[idx]) > strength_threshold * max_values[col] && row != col) {
                    sparsity_rows[row] += 1;
                }
            }
        }
    }
    for(int i=1; i < csr->get_size()[0] + 1; i++) {
        sparsity_rows[i] += sparsity_rows[i-1];
    }

}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_COMPUTE_STRONG_DEP_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void compute_strong_dep(std::shared_ptr<const DefaultExecutor> exec,
                        const matrix::Csr<ValueType, IndexType>* csr,
                        remove_complex<ValueType> strength_threshold,
                        matrix::SparsityCsr<ValueType, IndexType>* strong_dep)
{
    
    std::vector<IndexType> max_values(csr->get_size()[1]);

    for (IndexType col = 0; col < csr->get_size()[1]; ++col) {
        remove_complex<ValueType> max_val = 0;
        for (IndexType row = 0; row < csr->get_size()[0]; ++row) {
            for (auto idx = csr->get_const_row_ptrs()[row]; idx < csr->get_const_row_ptrs()[row + 1]; ++idx) {
                if (csr->get_const_col_idxs()[idx] == col && row != col) {
                    max_val = std::max(max_val, gko::abs(csr->get_const_values()[idx]));
                }
            }
        }
        max_values[col] = max_val;
    }

    for (IndexType col = 0; col < csr->get_size()[1]; ++col) {
        for (IndexType row = 0; row < csr->get_size()[0]; ++row) {
            for (auto idx = csr->get_const_row_ptrs()[row]; idx < csr->get_const_row_ptrs()[row + 1]; ++idx) {
                if (gko::abs(csr->get_const_values()[idx]) > strength_threshold * max_values[col] && row != col) {
                    strong_dep->get_col_idxs()[strong_dep->get_row_ptrs()[row]++] = col;
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_COMPUTE_STRONG_DEP_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_weight_and_status(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* strong_dep,
    remove_complex<ValueType>* weight, int* status)
{
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for(int i=0; i< strong_dep->get_num_nonzeros(); i++){
        weight[strong_dep->get_const_col_idxs()[i]] +=1;
    }
    for(int i=0; i<strong_dep->get_size()[0]; i++){
        if(weight[i] > 0){
            double rnd = dist(gen);
            weight[i] += static_cast<remove_complex<ValueType>>(rnd);
            status[i] = 0;
        }else{
            status[i] = 1;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_INITIALIZE_WEIGHT_AND_STATUS_KERNEL);


template <typename ValueType, typename IndexType>
void classify(std::shared_ptr<const DefaultExecutor> exec,
              const remove_complex<ValueType>* weight,
              const matrix::SparsityCsr<ValueType, IndexType>* strong_dep,
              int* status)
{
    GKO_NOT_IMPLEMENTED;
    
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PMIS_CLASSIFY_KERNEL);


void count(std::shared_ptr<const DefaultExecutor> exec,
           const array<int>& status, size_type* num)
{
    for(int i=0; i< status.get_size(); i++){
        if(status.get_const_data()[i] == 0){
            (*num)++;
        }
    }
}


}  // namespace pmis
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
