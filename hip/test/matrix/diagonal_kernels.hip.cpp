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

#include <ginkgo/core/matrix/diagonal.hpp>


#include <algorithm>
#include <numeric>
#include <random>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/matrix/diagonal_kernels.hpp"
#include "hip/test/utils.hip.hpp"


namespace {


class Diagonal : public ::testing::Test {
protected:
    using ValueType = double;
    using ComplexValueType = std::complex<double>;
    using Csr = gko::matrix::Csr<ValueType>;
    using Diag = gko::matrix::Diagonal<ValueType>;
    using Dense = gko::matrix::Dense<ValueType>;
    using ComplexDiag = gko::matrix::Diagonal<ComplexValueType>;

    Diagonal() : mtx_size(532, 231), rand_engine(42) {}

    void SetUp()
    {
        ASSERT_GT(gko::HipExecutor::get_num_devices(), 0);
        ref = gko::ReferenceExecutor::create();
        hip = gko::HipExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (hip != nullptr) {
            ASSERT_NO_THROW(hip->synchronize());
        }
    }

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols,
                                     int min_nnz_row)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(min_nnz_row, num_cols),
            std::normal_distribution<>(0.0, 1.0), rand_engine, ref);
    }

    std::unique_ptr<Diag> gen_diag(int size)
    {
        auto diag = Diag::create(ref, size);
        auto vals = diag->get_values();
        auto value_dist = std::normal_distribution<>(0.0, 1.0);
        for (int i = 0; i < size; i++) {
            vals[i] = gko::test::detail::get_rand_value<ValueType>(value_dist,
                                                                   rand_engine);
        }
        return diag;
    }

    std::unique_ptr<ComplexDiag> gen_cdiag(int size)
    {
        auto cdiag = ComplexDiag::create(ref, size);
        auto vals = cdiag->get_values();
        auto value_dist = std::normal_distribution<>(0.0, 1.0);
        for (int i = 0; i < size; i++) {
            vals[i] = ComplexValueType{
                gko::test::detail::get_rand_value<ComplexValueType>(
                    value_dist, rand_engine)};
        }
        return cdiag;
    }

    void set_up_apply_data()
    {
        diag = gen_diag(mtx_size[0]);
        ddiag = Diag::create(hip);
        ddiag->copy_from(diag.get());
        dense1 = gen_mtx<Dense>(mtx_size[0], mtx_size[1], mtx_size[0]);
        dense2 = gen_mtx<Dense>(mtx_size[1], mtx_size[0], mtx_size[1]);
        denseexpected1 = gen_mtx<Dense>(mtx_size[0], mtx_size[1], mtx_size[0]);
        denseexpected2 = gen_mtx<Dense>(mtx_size[1], mtx_size[0], mtx_size[1]);
        ddense1 = Dense::create(hip);
        ddense1->copy_from(dense1.get());
        ddense2 = Dense::create(hip);
        ddense2->copy_from(dense2.get());
        denseresult1 = Dense::create(hip);
        denseresult1->copy_from(denseexpected1.get());
        denseresult2 = Dense::create(hip);
        denseresult2->copy_from(denseexpected2.get());
        csr1 = gen_mtx<Csr>(mtx_size[0], mtx_size[1], 1);
        csr2 = gen_mtx<Csr>(mtx_size[1], mtx_size[0], 1);
        csrexpected1 = gen_mtx<Csr>(mtx_size[0], mtx_size[1], 1);
        csrexpected2 = gen_mtx<Csr>(mtx_size[1], mtx_size[0], 1);
        dcsr1 = Csr::create(hip);
        dcsr1->copy_from(csr1.get());
        dcsr2 = Csr::create(hip);
        dcsr2->copy_from(csr2.get());
        csrresult1 = Csr::create(hip);
        csrresult1->copy_from(csrexpected1.get());
        csrresult2 = Csr::create(hip);
        csrresult2->copy_from(csrexpected2.get());
    }

    void set_up_complex_data()
    {
        cdiag = gen_cdiag(mtx_size[0]);
        dcdiag = ComplexDiag::create(hip);
        dcdiag->copy_from(cdiag.get());
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::HipExecutor> hip;

    const gko::dim<2> mtx_size;
    std::ranlux48 rand_engine;

    std::unique_ptr<Diag> diag;
    std::unique_ptr<Diag> ddiag;
    std::unique_ptr<ComplexDiag> cdiag;
    std::unique_ptr<ComplexDiag> dcdiag;

    std::unique_ptr<Dense> dense1;
    std::unique_ptr<Dense> dense2;
    std::unique_ptr<Dense> denseexpected1;
    std::unique_ptr<Dense> denseexpected2;
    std::unique_ptr<Dense> denseresult1;
    std::unique_ptr<Dense> denseresult2;
    std::unique_ptr<Dense> ddense1;
    std::unique_ptr<Dense> ddense2;
    std::unique_ptr<Csr> csr1;
    std::unique_ptr<Csr> csr2;
    std::unique_ptr<Csr> dcsr1;
    std::unique_ptr<Csr> dcsr2;
    std::unique_ptr<Csr> csrexpected1;
    std::unique_ptr<Csr> csrexpected2;
    std::unique_ptr<Csr> csrresult1;
    std::unique_ptr<Csr> csrresult2;
};


TEST_F(Diagonal, ApplyToDenseIsEquivalentToRef)
{
    set_up_apply_data();

    diag->apply(dense1.get(), denseexpected1.get());
    ddiag->apply(ddense1.get(), denseresult1.get());

    GKO_ASSERT_MTX_NEAR(denseexpected1, denseresult1, 1e-14);
}


TEST_F(Diagonal, RightApplyToDenseIsEquivalentToRef)
{
    set_up_apply_data();

    diag->rapply(dense2.get(), denseexpected2.get());
    ddiag->rapply(ddense2.get(), denseresult2.get());

    GKO_ASSERT_MTX_NEAR(denseexpected2, denseresult2, 1e-14);
}


TEST_F(Diagonal, ApplyToCsrIsEquivalentToRef)
{
    set_up_apply_data();

    diag->apply(csr1.get(), csrexpected1.get());
    ddiag->apply(dcsr1.get(), csrresult1.get());

    GKO_ASSERT_MTX_NEAR(csrexpected1, csrresult1, 1e-14);
}


TEST_F(Diagonal, RightApplyToCsrIsEquivalentToRef)
{
    set_up_apply_data();

    diag->rapply(csr2.get(), csrexpected2.get());
    ddiag->rapply(dcsr2.get(), csrresult2.get());

    GKO_ASSERT_MTX_NEAR(csrexpected2, csrresult2, 1e-14);
}


TEST_F(Diagonal, ConvertToCsrIsEquivalentToRef)
{
    set_up_apply_data();

    diag->convert_to(csr1.get());
    ddiag->convert_to(dcsr1.get());

    GKO_ASSERT_MTX_NEAR(csr1, dcsr1, 0);
}


TEST_F(Diagonal, ConjTransposeIsEquivalentToRef)
{
    set_up_complex_data();

    auto trans = cdiag->conj_transpose();
    auto trans_diag = static_cast<ComplexDiag *>(trans.get());
    auto dtrans = dcdiag->conj_transpose();
    auto dtrans_diag = static_cast<ComplexDiag *>(dtrans.get());

    GKO_ASSERT_MTX_NEAR(trans_diag, dtrans_diag, 0);
}


TEST_F(Diagonal, InplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_data();

    diag->compute_absolute_inplace();
    ddiag->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(diag, ddiag, 1e-14);
}


TEST_F(Diagonal, OutplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_data();

    auto abs_diag = diag->compute_absolute();
    auto dabs_diag = ddiag->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_diag, dabs_diag, 1e-14);
}


}  // namespace
