#ifndef GKO_CORE_MATRIX_DENSE_KERNELS_HPP_
#define GKO_CORE_MATRIX_DENSE_KERNELS_HPP_


#include "core/base/types.hpp"
#include "core/matrix/dense.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_GEMM_KERNEL(_type)                    \
    void gemm(_type alpha, const matrix::Dense<_type> *a, \
              const matrix::Dense<_type> *b, _type beta,  \
              matrix::Dense<_type> *c)


namespace cpu {


template <typename ValueType>
GKO_DECLARE_GEMM_KERNEL(ValueType);


}  // namespace cpu


namespace gpu {


template <typename ValueType>
GKO_DECLARE_GEMM_KERNEL(ValueType);


}  // namespace gpu


namespace reference {


template <typename ValueType>
GKO_DECLARE_GEMM_KERNEL(ValueType);


}  // namespace reference
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_DENSE_KERNELS_HPP_
