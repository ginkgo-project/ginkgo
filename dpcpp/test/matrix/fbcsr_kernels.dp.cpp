// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "core/test/matrix/fbcsr_sample.hpp"
#include "core/test/utils.hpp"


namespace {


class Fbcsr : public ::testing::Test {
protected:
#if GINKGO_DPCPP_SINGLE_MODE
    using vtype = float;
#else
    using vtype = double;
#endif  // GINKGO_DPCPP_SINGLE_MODE
    using Mtx = gko::matrix::Fbcsr<vtype>;

    void SetUp()
    {
        ASSERT_GT(gko::DpcppExecutor::get_num_devices("all"), 0);
        ref = gko::ReferenceExecutor::create();
        dpcpp = gko::DpcppExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (dpcpp != nullptr) {
            ASSERT_NO_THROW(dpcpp->synchronize());
        }
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::DpcppExecutor> dpcpp;

    std::unique_ptr<Mtx> mtx;
};


TEST_F(Fbcsr, CanWriteFromMatrixOnDevice)
{
    using value_type = Mtx::value_type;
    using index_type = Mtx::index_type;
    using MatData = gko::matrix_data<value_type, index_type>;
    gko::testing::FbcsrSample<value_type, index_type> sample(ref);
    auto refmat = sample.generate_fbcsr();
    auto dpcppmat = gko::clone(dpcpp, refmat);
    MatData refdata;
    MatData dpcppdata;

    refmat->write(refdata);
    dpcppmat->write(dpcppdata);

    ASSERT_TRUE(refdata.nonzeros == dpcppdata.nonzeros);
}


}  // namespace
