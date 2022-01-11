/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/matrix/dense.hpp>


#include <algorithm>
#include <numeric>
#include <random>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>


#include "core/components/fill_array_kernels.hpp"
#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


namespace {


class Identity : public ::testing::Test {
protected:
    using itype = int;
#if GINKGO_COMMON_SINGLE_MODE
    using vtype = float;
#else
    using vtype = double;
#endif
    using Id = gko::matrix::Identity<vtype>;
    using Dense = gko::matrix::Dense<vtype>;
    using Arr = gko::Array<itype>;
    using ComplexMtx = gko::matrix::Dense<std::complex<vtype>>;

    Identity() : rand_engine(15) {}

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        init_executor(ref, exec);
    }

    void TearDown()
    {
        if (exec != nullptr) {
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols,
                                     int min_nnz_row)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(min_nnz_row, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void set_up_apply_data()
    {
        id = Id::create(ref, sz);
        alpha = gko::initialize<Dense>({2.0}, ref);
        beta = gko::initialize<Dense>({-1.0}, ref);
        did = Id::create(exec, sz);
        dalpha = gko::clone(exec, alpha);
        dbeta = gko::clone(exec, beta);
    }

    template <typename ConvertedType, typename InputType>
    std::unique_ptr<ConvertedType> convert(InputType&& input)
    {
        auto result = ConvertedType::create(input->get_executor());
        input->convert_to(result.get());
        return result;
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::EXEC_TYPE> exec;

    std::ranlux48 rand_engine;

	gko::size_type sz = 41;
    std::unique_ptr<Id> id;
    std::unique_ptr<Dense> alpha;
    std::unique_ptr<Dense> beta;
    std::unique_ptr<Id> did;
    std::unique_ptr<Dense> dalpha;
    std::unique_ptr<Dense> dbeta;
};


TEST_F(Identity, AddScaledIdentityToDense)
{
    set_up_apply_data();
    auto x = gen_mtx<Dense>(sz, sz, sz-5);
    auto dx = gko::clone(exec, x);

    id->apply(alpha.get(), id.get(), beta.get(), x.get());
    did->apply(dalpha.get(), did.get(), dbeta.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(x, dx, r<vtype>::value);
}


TEST_F(Identity, AddScaledIdentityToCsr)
{
    using Csr = gko::matrix::Csr<vtype, itype>;
    set_up_apply_data();
    auto mtx = gen_mtx<Csr>(id->get_size()[0], id->get_size()[1], 5);
    auto dmtx = gko::clone(exec, mtx);
    dmtx->copy_from(mtx.get());

    id->apply(alpha.get(), id.get(), beta.get(), mtx.get());
    did->apply(dalpha.get(), did.get(), dbeta.get(), dmtx.get());

    GKO_ASSERT_MTX_NEAR(mtx, dmtx, r<vtype>::value);
}


}
