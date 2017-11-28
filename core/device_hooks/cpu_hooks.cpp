#include "core/base/exception.hpp"
#include "core/base/exception_helpers.hpp"
#include "core/matrix/dense_kernels.hpp"


namespace gko {


namespace kernels {


namespace cpu {


template <typename ValueType>
GKO_DECLARE_GEMM_KERNEL(ValueType)
NOT_COMPILED(cpu);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GEMM_KERNEL);

template <typename ValueType>
GKO_DECLARE_SCAL_KERNEL(ValueType)
NOT_COMPILED(cpu);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_SCAL_KERNEL);

template <typename ValueType>
GKO_DECLARE_AXPY_KERNEL(ValueType)
NOT_COMPILED(cpu);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_AXPY_KERNEL);

template <typename ValueType>
GKO_DECLARE_DOT_KERNEL(ValueType)
NOT_COMPILED(cpu);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DOT_KERNEL);


}  // namespace cpu


}  // namespace kernels


}  // namespace gko
