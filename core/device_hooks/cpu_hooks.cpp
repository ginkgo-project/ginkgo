#include "core/base/exception.hpp"
#include "core/base/exception_helpers.hpp"
#include "core/matrix/dense.hpp"


namespace gko {


namespace kernels {


namespace cpu {


template <typename ValueType>
GINKGO_DECLARE_GEMM_KERNEL(ValueType)
NOT_COMPILED(cpu);
GINKGO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GINKGO_DECLARE_GEMM_KERNEL);


}  // namespace cpu


}  // namespace kernels


}  // namespace gko
