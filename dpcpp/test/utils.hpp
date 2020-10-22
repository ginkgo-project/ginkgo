#ifndef GKO_DPCPP_TEST_UTILS_HPP_
#define GKO_DPCPP_TEST_UTILS_HPP_


#include "core/test/utils.hpp"


#include <ginkgo/core/base/executor.hpp>


namespace {


// prevent device reset after each test
auto no_reset_exec =
    gko::DpcppExecutor::create(0, gko::ReferenceExecutor::create());


}  // namespace


#endif  // GKO_DPCPP_TEST_UTILS_HPP_
