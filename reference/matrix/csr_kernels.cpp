#include "core/matrix/csr_kernels.hpp"


#include "core/base/exception_helpers.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace csr {


template <typename ValueType, typename IndexType>
void spmv(const matrix::Csr<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    ASSERT_CONFORMANT(a, b);
    ASSERT_EQUAL_ROWS(a, c);
    ASSERT_EQUAL_COLS(b, c);
    auto row_ptrs = a->get_const_row_ptrs();
    auto col_idxs = a->get_const_col_idxs();
    auto vals = a->get_const_values();

    for (size_type row = 0; row < a->get_num_rows(); ++row) {
        for (size_type j = 0; j < c->get_num_cols(); ++j) {
            c->at(row, j) = zero<ValueType>();
        }
        for (size_type k = row_ptrs[row]; k < row_ptrs[row + 1]; ++k) {
            auto val = vals[k];
            auto col = col_idxs[k];
            for (size_type j = 0; j < c->get_num_cols(); ++j) {
                c->at(row, j) += val * b->at(col, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(const matrix::Dense<ValueType> *alpha,
                   const matrix::Csr<ValueType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   const matrix::Dense<ValueType> *beta,
                   matrix::Dense<ValueType> *c)
{
    ASSERT_CONFORMANT(a, b);
    ASSERT_EQUAL_ROWS(a, c);
    ASSERT_EQUAL_COLS(b, c);
    auto row_ptrs = a->get_const_row_ptrs();
    auto col_idxs = a->get_const_col_idxs();
    auto vals = a->get_const_values();
    auto valpha = alpha->at(0, 0);
    auto vbeta = beta->at(0, 0);

    for (size_type row = 0; row < a->get_num_rows(); ++row) {
        for (size_type j = 0; j < c->get_num_cols(); ++j) {
            c->at(row, j) *= vbeta;
        }
        for (size_type k = row_ptrs[row]; k < row_ptrs[row + 1]; ++k) {
            auto val = vals[k];
            auto col = col_idxs[k];
            for (size_type j = 0; j < c->get_num_cols(); ++j) {
                c->at(row, j) += valpha * val * b->at(col, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ADVANCED_SPMV_KERNEL);


}  // namespace csr
}  // namespace reference
}  // namespace kernels
}  // namespace gko
