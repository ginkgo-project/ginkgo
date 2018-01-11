#ifndef GKO_CORE_MATRIX_DENSE_KERNELS_HPP_
#define GKO_CORE_MATRIX_DENSE_KERNELS_HPP_


#include "core/base/types.hpp"
#include "core/matrix/dense.hpp"


namespace gko {
namespace kernels {

#define GKO_DECLARE_DENSE_SIMPLE_APPLY_KERNEL(_type) \
    void simple_apply(const matrix::Dense<_type> *a, \
                      const matrix::Dense<_type> *b, matrix::Dense<_type> *c)

#define GKO_DECLARE_DENSE_APPLY_KERNEL(_type)                                \
    void apply(const matrix::Dense<_type> *alpha,                            \
               const matrix::Dense<_type> *a, const matrix::Dense<_type> *b, \
               const matrix::Dense<_type> *beta, matrix::Dense<_type> *c)


#define GKO_DECLARE_DENSE_SCALE_KERNEL(_type) \
    void scale(const matrix::Dense<_type> *alpha, matrix::Dense<_type> *x)


#define GKO_DECLARE_DENSE_ADD_SCALED_KERNEL(_type)     \
    void add_scaled(const matrix::Dense<_type> *alpha, \
                    const matrix::Dense<_type> *x, matrix::Dense<_type> *y)


#define GKO_DECLARE_DENSE_COMPUTE_DOT_KERNEL(_type) \
    void compute_dot(const matrix::Dense<_type> *x, \
                     const matrix::Dense<_type> *y, \
                     matrix::Dense<_type> *result)


#define DECLARE_ALL_AS_TEMPLATES                      \
    template <typename ValueType>                     \
    GKO_DECLARE_DENSE_SIMPLE_APPLY_KERNEL(ValueType); \
    template <typename ValueType>                     \
    GKO_DECLARE_DENSE_APPLY_KERNEL(ValueType);        \
    template <typename ValueType>                     \
    GKO_DECLARE_DENSE_SCALE_KERNEL(ValueType);        \
    template <typename ValueType>                     \
    GKO_DECLARE_DENSE_ADD_SCALED_KERNEL(ValueType);   \
    template <typename ValueType>                     \
    GKO_DECLARE_DENSE_COMPUTE_DOT_KERNEL(ValueType)


namespace cpu {
namespace dense {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace dense
}  // namespace cpu


namespace gpu {
namespace dense {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace dense
}  // namespace gpu


namespace reference {
namespace dense {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace dense
}  // namespace reference
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_DENSE_KERNELS_HPP_
