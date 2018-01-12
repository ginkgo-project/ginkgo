#include "core/base/exception_helpers.hpp"
#include "core/matrix/dense_kernels.hpp"


#ifndef GKO_HOOK_MODULE
#error "Need to define GKO_HOOK_MODULE variable before including this file"
#endif  // GKO_HOOK_MODULE


namespace gko {
namespace kernels {
namespace GKO_HOOK_MODULE {
namespace dense {


template <typename ValueType>
GKO_DECLARE_DENSE_SIMPLE_APPLY_KERNEL(ValueType)
NOT_COMPILED(GKO_HOOK_MODULE);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_SIMPLE_APPLY_KERNEL);

template <typename ValueType>
GKO_DECLARE_DENSE_APPLY_KERNEL(ValueType)
NOT_COMPILED(GKO_HOOK_MODULE);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_APPLY_KERNEL);

template <typename ValueType>
GKO_DECLARE_DENSE_SCALE_KERNEL(ValueType)
NOT_COMPILED(GKO_HOOK_MODULE);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_SCALE_KERNEL);

template <typename ValueType>
GKO_DECLARE_DENSE_ADD_SCALED_KERNEL(ValueType)
NOT_COMPILED(GKO_HOOK_MODULE);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_ADD_SCALED_KERNEL);

template <typename ValueType>
GKO_DECLARE_DENSE_COMPUTE_DOT_KERNEL(ValueType)
NOT_COMPILED(GKO_HOOK_MODULE);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_DOT_KERNEL);


}  // namespace dense
}  // namespace GKO_HOOK_MODULE
}  // namespace kernels
}  // namespace gko
