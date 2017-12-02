#ifndef GKO_CORE_MATRIX_CSR_KERNELS_HPP_
#define GKO_CORE_MATRIX_CSR_KERNELS_HPP_


#include "core/matrix/csr.hpp"
#include "core/matrix/dense.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_CSR_SPMV_KERNEL(ValueType, IndexType) \
    void spmv(const matrix::Csr<ValueType, IndexType> *a, \
              const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)

#define GKO_DECLARE_CSR_ADVANCED_SPMV_KERNEL(ValueType, IndexType) \
    void advanced_spmv(const matrix::Dense<ValueType> *alpha,      \
                       const matrix::Csr<ValueType, IndexType> *a, \
                       const matrix::Dense<ValueType> *b,          \
                       const matrix::Dense<ValueType> *beta,       \
                       matrix::Dense<ValueType> *c)


#define DECLARE_ALL_AS_TEMPLATES                       \
    template <typename ValueType, typename IndexType>  \
    GKO_DECLARE_CSR_SPMV_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>  \
    GKO_DECLARE_CSR_ADVANCED_SPMV_KERNEL(ValueType, IndexType)


namespace cpu {
namespace csr {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace csr
}  // namespace cpu


namespace gpu {
namespace csr {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace csr
}  // namespace gpu


namespace reference {
namespace csr {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace csr
}  // namespace reference


#undef DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_CSR_KERNELS_HPP_
