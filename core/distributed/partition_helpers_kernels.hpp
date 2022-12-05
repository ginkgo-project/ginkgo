#ifndef GINKGO_PARTITION_HELPERS_KERNELS_HPP
#define GINKGO_PARTITION_HELPERS_KERNELS_HPP


#include <ginkgo/core/base/array.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_PARTITION_HELPERS_COMPRESS_START_ENDS(_type)          \
    void compress_start_ends(std::shared_ptr<const DefaultExecutor> exec, \
                             const array<_type>& range_start_ends,        \
                             array<_type>& ranges)


#define GKO_DECLARE_ALL_AS_TEMPLATES    \
    template <typename GlobalIndexType> \
    GKO_DECLARE_PARTITION_HELPERS_COMPRESS_START_ENDS(GlobalIndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(partition_helpers,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GINKGO_PARTITION_HELPERS_KERNELS_HPP
