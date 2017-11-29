#ifndef GKO_CORE_MATRIX_DENSE_KERNELS_HPP_
#define GKO_CORE_MATRIX_DENSE_KERNELS_HPP_


#include "core/base/types.hpp"
#include "core/matrix/dense.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_GEMM_KERNEL(_type)                                      \
    void gemm(const matrix::Dense<_type> *alpha,                            \
              const matrix::Dense<_type> *a, const matrix::Dense<_type> *b, \
              const matrix::Dense<_type> *beta, matrix::Dense<_type> *c)


#define GKO_DECLARE_SCAL_KERNEL(_type) \
    void scal(const matrix::Dense<_type> *alpha, matrix::Dense<_type> *x)


#define GKO_DECLARE_AXPY_KERNEL(_type)           \
    void axpy(const matrix::Dense<_type> *alpha, \
              const matrix::Dense<_type> *x, matrix::Dense<_type> *y)


#define GKO_DECLARE_DOT_KERNEL(_type)                                      \
    void dot(const matrix::Dense<_type> *x, const matrix::Dense<_type> *y, \
             matrix::Dense<_type> *result)


#define DECLARE_ALL_AS_TEMPLATES        \
    template <typename ValueType>       \
    GKO_DECLARE_GEMM_KERNEL(ValueType); \
    template <typename ValueType>       \
    GKO_DECLARE_SCAL_KERNEL(ValueType); \
    template <typename ValueType>       \
    GKO_DECLARE_AXPY_KERNEL(ValueType); \
    template <typename ValueType>       \
    GKO_DECLARE_DOT_KERNEL(ValueType)


namespace cpu {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace cpu


namespace gpu {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace gpu


namespace reference {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace reference
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_DENSE_KERNELS_HPP_
