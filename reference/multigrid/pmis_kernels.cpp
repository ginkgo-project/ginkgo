// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/pmis_kernels.hpp"

#include <algorithm>
#include <memory>
#include <tuple>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include <random>

namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The PMIS solver namespace.
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
    using rc = remove_complex<ValueType>;

    const auto nrow = csr->get_size()[0]; 
    const auto row_ptrs = csr->get_const_row_ptrs();   
    const auto col_idxs = csr->get_const_col_idxs();   
    const auto vals     = csr->get_const_values();  

    sparsity_rows[0] = IndexType{0};

    for (auto i = 0; i < nrow; ++i) {
        
        sparsity_rows[i+1] = sparsity_rows[i];

        const auto row_start = row_ptrs[i];
        const auto row_end   = row_ptrs[i + 1];

        for (auto j = row_start; j < row_end; ++j) {
            const auto k = col_idxs[j];
            if (k == i) continue; 

            rc max_abs = rc{0};
            
            for (auto r = 0; r < nrow; ++r) {
                const auto start = row_ptrs[r];
                const auto end = row_ptrs[r+1];
                for (auto jj = start; jj < end; ++jj) {
                    if (col_idxs[jj] == k && r != k) {
                        const rc a = gko::abs(vals[jj]);
                        if (a > max_abs) 
                            max_abs = a;
                    }
                }
            }

            if (max_abs > rc{0} && gko::abs(vals[j]) >= strength_threshold * max_abs) {
                sparsity_rows[i+1]++;
            }
        }
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
    std::vector<remove_complex<ValueType>> max_values(csr->get_size()[1]);

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

std::vector<IndexType> row_offsets(csr->get_size()[0]);
for (IndexType row = 0; row < csr->get_size()[0]; ++row) {
    row_offsets[row] = strong_dep->get_row_ptrs()[row];
}

    for (IndexType col = 0; col < csr->get_size()[1]; ++col) {
        for (IndexType row = 0; row < csr->get_size()[0]; ++row) {
            for (auto idx = csr->get_const_row_ptrs()[row]; idx < csr->get_const_row_ptrs()[row + 1]; ++idx) {
                if (csr->get_const_col_idxs()[idx] == col && gko::abs(csr->get_const_values()[idx]) >= strength_threshold * max_values[col] && row != col) {
                    strong_dep->get_col_idxs()[row_offsets[row]++] = col;
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

    using rc = remove_complex<ValueType>;

    const auto nrows = static_cast<IndexType>(strong_dep->get_size()[0]);
    const auto s_row_ptrs = strong_dep->get_const_row_ptrs(); 
    const auto s_col_idxs = strong_dep->get_const_col_idxs(); 

    for (auto r = 0; r < nrows; ++r)
        weight[r] = rc{0};
    
    for (auto r = 0; r < nrows; ++r) {
        for (auto p = s_row_ptrs[r]; p < s_row_ptrs[r + 1]; ++p) {
            auto c = s_col_idxs[p];
            weight[c] += rc{1};
        }
    }
    for (auto i = 0; i < nrows; ++i) {
        status[i] = (weight[i] == 0 ? 1 : 0); 
        weight[i] += static_cast<rc>(dist(gen));
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
    const auto nrows = static_cast<IndexType>(strong_dep->get_size()[0]);
    const auto s_row_ptrs = strong_dep->get_const_row_ptrs();
    const auto s_col_idxs = strong_dep->get_const_col_idxs();
    std::vector<char> higher(nrows, 0);

    auto trans_l = strong_dep->transpose(); 
    auto transposed_sparsity = gko::as<matrix::SparsityCsr<ValueType, IndexType>>(trans_l.get());
    const auto tr_row_ptrs = transposed_sparsity->get_const_row_ptrs();
    const auto tr_col_idxs = transposed_sparsity->get_const_col_idxs();

    for (IndexType i=0; i < nrows; i++){
        if (status[i] == 0) {
            char check = 'c';

            const auto row_start = s_row_ptrs[i];
            const auto row_end = s_row_ptrs[i + 1];

            for (IndexType j = row_start; j < row_end; ++j) {
                auto c = s_col_idxs[j];
                if (status[c] == 0 && weight[i] == weight[c]) {
                    check = 'a';
                }
                if (status[c] == 0 && weight[i] < weight[c]) {
                    check = 'b';
                    break;
                }
            }
            if (check == 'c' || check == 'a') {
                const auto tr_row_start = tr_row_ptrs[i];
                const auto tr_row_end = tr_row_ptrs[i + 1];
                for (IndexType j = tr_row_start; j < tr_row_end; ++j) {
                    auto c = tr_col_idxs[j];
                    if (status[c] == 0 && weight[i] == weight[c]) {
                        check = 'a';
                    }
                    if (status[c] == 0 && weight[i] < weight[c]) {
                        check = 'b';
                        break;
                    }
                }
            }
            if (check == 'c' || check == 'a') {
                status[i] = 2;
                const auto row_start = tr_row_ptrs[i];
                const auto row_end = tr_row_ptrs[i + 1];
                for (IndexType j= row_start; j < row_end; ++j) {
                    auto c = tr_col_idxs[j];
                    if (status[c] == 0) 
                        status[c] = 1; 
                }
            }   
        }
    }
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
}  // namespace reference
}  // namespace kernels
}  // namespace gko
