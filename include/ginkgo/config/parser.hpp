#include <ginkgo/core/base/lin_op.hpp>
namespace gko {
namespace config {


template <typename ValueType = double, typename IndexType = gko::int32>
std::shared_ptr<const LinOpFactory> parse_config(
    std::istream& stream, std::shared_ptr<const Executor> exec);


}
}  // namespace gko