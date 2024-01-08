// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/diagonal_kernels.hpp"


#include <algorithm>
#include <numeric>
#include <random>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


class Diagonal : public CommonTestFixture {
protected:
    using ValueType = value_type;
    using ComplexValueType = std::complex<value_type>;
    using Csr = gko::matrix::Csr<ValueType>;
    using Diag = gko::matrix::Diagonal<ValueType>;
    using Dense = gko::matrix::Dense<ValueType>;
    using ComplexDiag = gko::matrix::Diagonal<ComplexValueType>;

    Diagonal()
#ifdef GINKGO_FAST_TESTS
        : mtx_size(152, 231),
#else
        : mtx_size(532, 231),
#endif
          rand_engine(42)
    {}

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols,
                                     int min_nnz_row)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(min_nnz_row, num_cols),
            std::normal_distribution<value_type>(0.0, 1.0), rand_engine, ref);
    }

    std::unique_ptr<Diag> gen_diag(int size)
    {
        auto diag = Diag::create(ref, size);
        auto vals = diag->get_values();
        auto value_dist = std::normal_distribution<value_type>(0.0, 1.0);
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
        auto value_dist = std::normal_distribution<value_type>(0.0, 1.0);
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
        ddiag = gko::clone(exec, diag);
        dense1 = gen_mtx<Dense>(mtx_size[0], mtx_size[1], mtx_size[1]);
        dense2 = gen_mtx<Dense>(mtx_size[1], mtx_size[0], mtx_size[0]);
        denseexpected1 = gen_mtx<Dense>(mtx_size[0], mtx_size[1], mtx_size[1]);
        denseexpected2 = gen_mtx<Dense>(mtx_size[1], mtx_size[0], mtx_size[0]);
        ddense1 = gko::clone(exec, dense1);
        ddense2 = gko::clone(exec, dense2);
        denseresult1 = gko::clone(exec, denseexpected1);
        denseresult2 = gko::clone(exec, denseexpected2);
        csr1 = gen_mtx<Csr>(mtx_size[0], mtx_size[1], 1);
        csr2 = gen_mtx<Csr>(mtx_size[1], mtx_size[0], 1);
        csrexpected1 = gen_mtx<Csr>(mtx_size[0], mtx_size[1], 1);
        csrexpected2 = gen_mtx<Csr>(mtx_size[1], mtx_size[0], 1);
        dcsr1 = gko::clone(exec, csr1);
        dcsr2 = gko::clone(exec, csr2);
        csrresult1 = gko::clone(exec, csrexpected1);
        csrresult2 = gko::clone(exec, csrexpected2);
    }

    void set_up_complex_data()
    {
        cdiag = gen_cdiag(mtx_size[0]);
        dcdiag = gko::clone(exec, cdiag);
    }

    const gko::dim<2> mtx_size;
    std::default_random_engine rand_engine;

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

    diag->apply(dense1, denseexpected1);
    ddiag->apply(ddense1, denseresult1);

    GKO_ASSERT_MTX_NEAR(denseexpected1, denseresult1, r<value_type>::value);
}


TEST_F(Diagonal, RightApplyToDenseIsEquivalentToRef)
{
    set_up_apply_data();

    diag->rapply(dense2, denseexpected2);
    ddiag->rapply(ddense2, denseresult2);

    GKO_ASSERT_MTX_NEAR(denseexpected2, denseresult2, r<value_type>::value);
}


TEST_F(Diagonal, InverseApplyToDenseIsEquivalentToRef)
{
    set_up_apply_data();

    diag->inverse_apply(dense1, denseexpected1);
    ddiag->inverse_apply(ddense1, denseresult1);

    GKO_ASSERT_MTX_NEAR(denseexpected2, denseresult2, r<value_type>::value);
}


TEST_F(Diagonal, ApplyToCsrIsEquivalentToRef)
{
    set_up_apply_data();

    diag->apply(csr1, csrexpected1);
    ddiag->apply(dcsr1, csrresult1);

    GKO_ASSERT_MTX_NEAR(csrexpected1, csrresult1, r<value_type>::value);
}


TEST_F(Diagonal, RightApplyToCsrIsEquivalentToRef)
{
    set_up_apply_data();

    diag->rapply(csr2, csrexpected2);
    ddiag->rapply(dcsr2, csrresult2);

    GKO_ASSERT_MTX_NEAR(csrexpected2, csrresult2, r<value_type>::value);
}


TEST_F(Diagonal, InverseApplyToCsrIsEquivalentToRef)
{
    set_up_apply_data();

    diag->inverse_apply(csr1, csrexpected1);
    ddiag->inverse_apply(dcsr1, csrresult1);

    GKO_ASSERT_MTX_NEAR(csrexpected2, csrresult2, r<value_type>::value);
}


TEST_F(Diagonal, ConvertToCsrIsEquivalentToRef)
{
    set_up_apply_data();

    diag->convert_to(csr1);
    ddiag->convert_to(dcsr1);

    GKO_ASSERT_MTX_NEAR(csr1, dcsr1, 0);
}


TEST_F(Diagonal, ConjTransposeIsEquivalentToRef)
{
    set_up_complex_data();

    auto trans = cdiag->conj_transpose();
    auto trans_diag = static_cast<ComplexDiag*>(trans.get());
    auto dtrans = dcdiag->conj_transpose();
    auto dtrans_diag = static_cast<ComplexDiag*>(dtrans.get());

    GKO_ASSERT_MTX_NEAR(trans_diag, dtrans_diag, 0);
}


TEST_F(Diagonal, InplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_data();

    diag->compute_absolute_inplace();
    ddiag->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(diag, ddiag, r<value_type>::value);
}


TEST_F(Diagonal, OutplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_data();

    auto abs_diag = diag->compute_absolute();
    auto dabs_diag = ddiag->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_diag, dabs_diag, r<value_type>::value);
}
