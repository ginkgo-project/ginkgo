#ifndef GINKGO_ENVIRONMENTS_HPP
#define GINKGO_ENVIRONMENTS_HPP


#include <ginkgo/core/base/exception_helpers.hpp>


#ifdef GKO_COMPILING_CUDA

#include "cuda/base/device.hpp"

class CudaEnvironment : public ::testing::Environment {
public:
    void TearDown() override { gko::kernels::cuda::reset_device(0); }
};

#else

class CudaEnvironment : public ::testing::Environment {};

#endif


#ifdef GKO_COMPILING_HIP

#include "hip/base/device.hpp"

class HipEnvironment : public ::testing::Environment {
public:
    void TearDown() override { gko::kernels::hip::reset_device(0); }
};

#else

class HipEnvironment : public ::testing::Environment {};

#endif


#endif  // GINKGO_ENVIRONMENTS_HPP
