#include "core/matrix/csr_kernels.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace csr {


template <typename ValueType, typename IndexType>
void spmv(const matrix::Csr<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    // TODO
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(const matrix::Dense<ValueType> *alpha,
                   const matrix::Csr<ValueType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   const matrix::Dense<ValueType> *beta,
                   matrix::Dense<ValueType> *c)
{
    // TODO
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ADVANCED_SPMV_KERNEL);


}  // namespace csr
}  // namespace reference
}  // namespace kernels
}  // namespace gko
