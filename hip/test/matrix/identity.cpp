/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <ginkgo/core/matrix/identity.hpp>


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"


namespace {


class Identity : public ::testing::Test {
protected:
    using itype = int;
    using vtype = double;
    using Id = gko::matrix::Identity<vtype>;
    using Mtx = gko::matrix::Dense<vtype>;
    using Csr = gko::matrix::Csr<vtype, itype>;

    Identity() : rand_engine(15) {}

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        dexec = gko::HipExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (dexec != nullptr) {
            ASSERT_NO_THROW(dexec->synchronize());
        }
    }

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(0.0, 1.0), rand_engine, ref);
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
        id = Id::create(ref, 41);
        x = gen_mtx<Mtx>(41, 41);
        alpha = gko::initialize<Mtx>({2.0}, ref);
        beta = gko::initialize<Mtx>({-1.0}, ref);
        did = Id::create(dexec, 41);
        dx = gko::clone(dexec, x);
        dalpha = gko::clone(dexec, alpha);
        dbeta = gko::clone(dexec, beta);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::HipExecutor> dexec;

    std::ranlux48 rand_engine;

    std::unique_ptr<Id> id;
    std::unique_ptr<Mtx> x;
    std::unique_ptr<Mtx> alpha;
    std::unique_ptr<Mtx> beta;
    std::unique_ptr<Id> did;
    std::unique_ptr<Mtx> dx;
    std::unique_ptr<Mtx> dalpha;
    std::unique_ptr<Mtx> dbeta;
};


TEST_F(Identity, ScaleDenseAddIdentity)
{
    set_up_apply_data();

    id->apply(alpha.get(), id.get(), beta.get(), x.get());
    did->apply(dalpha.get(), did.get(), dbeta.get(), dx.get());

    GKO_ASSERT_MTX_NEAR(x, dx, r<vtype>::value);
}


TEST_F(Identity, ScaleCsrAddIdentity)
{
    set_up_apply_data();
    auto mtx = gen_mtx<Csr>(id->get_size()[0], id->get_size()[1], 5);
    auto dmtx = Csr::create(dexec);
    dmtx->copy_from(mtx.get());

    id->apply(alpha.get(), id.get(), beta.get(), mtx.get());
    did->apply(dalpha.get(), did.get(), dbeta.get(), dmtx.get());

    GKO_ASSERT_MTX_NEAR(mtx, dmtx, r<vtype>::value);
}


}  // namespace
