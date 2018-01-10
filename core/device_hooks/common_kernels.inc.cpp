#include "core/base/exception_helpers.hpp"
#include "core/matrix/dense_kernels.hpp"


#ifndef GKO_HOOK_MODULE
#error "Need to define GKO_HOOK_MODULE variable before including this file"
#endif  // GKO_HOOK_MODULE


namespace gko {
namespace kernels {
namespace GKO_HOOK_MODULE {


template <typename ValueType>
GKO_DECLARE_GEMM_KERNEL(ValueType)
NOT_COMPILED(GKO_HOOK_MODULE);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GEMM_KERNEL);

template <typename ValueType>
GKO_DECLARE_SCAL_KERNEL(ValueType)
NOT_COMPILED(GKO_HOOK_MODULE);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_SCAL_KERNEL);

template <typename ValueType>
GKO_DECLARE_AXPY_KERNEL(ValueType)
NOT_COMPILED(GKO_HOOK_MODULE);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_AXPY_KERNEL);

template <typename ValueType>
GKO_DECLARE_DOT_KERNEL(ValueType)
NOT_COMPILED(GKO_HOOK_MODULE);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DOT_KERNEL);


}  // namespace GKO_HOOK_MODULE
}  // namespace kernels
}  // namespace gko
