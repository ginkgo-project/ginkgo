/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2019

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/sellp.hpp>


#include "core/matrix/dense_kernels.hpp"


#include <gtest/gtest.h>


#include <random>


#include <core/test/utils.hpp>


namespace {


class Dense : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;
    using ComplexMtx = gko::matrix::Dense<std::complex<double>>;

    Dense() : rand_engine(15) {}

    void SetUp()
    {
        ASSERT_GT(gko::CudaExecutor::get_num_devices(), 0);
        ref = gko::ReferenceExecutor::create();
        cuda = gko::CudaExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (cuda != nullptr) {
            ASSERT_NO_THROW(cuda->synchronize());
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

    void set_up_vector_data(gko::size_type num_vecs,
                            bool different_alpha = false)
    {
        x = gen_mtx<Mtx>(1000, num_vecs);
        y = gen_mtx<Mtx>(1000, num_vecs);
        if (different_alpha) {
            alpha = gen_mtx<Mtx>(1, num_vecs);
        } else {
            alpha = gko::initialize<Mtx>({2.0}, ref);
        }
        dx = Mtx::create(cuda);
        dx->copy_from(x.get());
        dy = Mtx::create(cuda);
        dy->copy_from(y.get());
        dalpha = Mtx::create(cuda);
        dalpha->copy_from(alpha.get());
        expected = Mtx::create(ref, gko::dim<2>{1, num_vecs});
        dresult = Mtx::create(cuda, gko::dim<2>{1, num_vecs});
    }

    void set_up_apply_data()
    {
        x = gen_mtx<Mtx>(65, 25);
        c_x = gen_mtx<ComplexMtx>(65, 25);
        y = gen_mtx<Mtx>(25, 35);
        expected = gen_mtx<Mtx>(65, 35);
        alpha = gko::initialize<Mtx>({2.0}, ref);
        beta = gko::initialize<Mtx>({-1.0}, ref);
        dx = Mtx::create(cuda);
        dx->copy_from(x.get());
        dc_x = ComplexMtx::create(cuda);
        dc_x->copy_from(c_x.get());
        dy = Mtx::create(cuda);
        dy->copy_from(y.get());
        dresult = Mtx::create(cuda);
        dresult->copy_from(expected.get());
        dalpha = Mtx::create(cuda);
        dalpha->copy_from(alpha.get());
        dbeta = Mtx::create(cuda);
        dbeta->copy_from(beta.get());
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::CudaExecutor> cuda;

    std::ranlux48 rand_engine;

    std::unique_ptr<Mtx> x;
    std::unique_ptr<ComplexMtx> c_x;
    std::unique_ptr<Mtx> y;
    std::unique_ptr<Mtx> alpha;
    std::unique_ptr<Mtx> beta;
    std::unique_ptr<Mtx> expected;
    std::unique_ptr<Mtx> dresult;
    std::unique_ptr<Mtx> dx;
    std::unique_ptr<ComplexMtx> dc_x;
    std::unique_ptr<Mtx> dy;
    std::unique_ptr<Mtx> dalpha;
    std::unique_ptr<Mtx> dbeta;
};


TEST_F(Dense, SingleVectorCudaScaleIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->scale(alpha.get());
    dx->scale(dalpha.get());

    auto result = Mtx::create(ref);
    result->copy_from(dx.get());
    GKO_ASSERT_MTX_NEAR(result, x, 1e-14);
}


TEST_F(Dense, MultipleVectorCudaScaleIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->scale(alpha.get());
    dx->scale(dalpha.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(Dense, MultipleVectorCudaScaleWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    x->scale(alpha.get());
    dx->scale(dalpha.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(Dense, SingleVectorCudaAddScaledIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->add_scaled(alpha.get(), y.get());
    dx->add_scaled(dalpha.get(), dy.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(Dense, MultipleVectorCudaAddScaledIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->add_scaled(alpha.get(), y.get());
    dx->add_scaled(dalpha.get(), dy.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(Dense, MultipleVectorCudaAddScaledWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->add_scaled(alpha.get(), y.get());
    dx->add_scaled(dalpha.get(), dy.get());

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(Dense, SingleVectorCudaComputeDotIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->compute_dot(y.get(), expected.get());
    dx->compute_dot(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Dense, MultipleVectorCudaComputeDotIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->compute_dot(y.get(), expected.get());
    dx->compute_dot(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Dense, CudaComputeNorm2IsEquivalentToRef)
{
    set_up_vector_data(20);

    x->compute_norm2(expected.get());
    dx->compute_norm2(dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Dense, SimpleApplyIsEquivalentToRef)
{
    set_up_apply_data();

    x->apply(y.get(), expected.get());
    dx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Dense, AdvancedApplyIsEquivalentToRef)
{
    set_up_apply_data();

    x->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Dense, IsTransposable)
{
    set_up_apply_data();

    auto trans = x->transpose();
    auto dtrans = dx->transpose();

    GKO_ASSERT_MTX_NEAR(static_cast<Mtx *>(dtrans.get()),
                        static_cast<Mtx *>(trans.get()), 0);
}


TEST_F(Dense, IsConjugateTransposable)
{
    set_up_apply_data();

    auto trans = c_x->conj_transpose();
    auto dtrans = dc_x->conj_transpose();

    GKO_ASSERT_MTX_NEAR(static_cast<ComplexMtx *>(dtrans.get()),
                        static_cast<ComplexMtx *>(trans.get()), 0);
}


TEST_F(Dense, ConvertToCooIsEquivalentToRef)
{
    set_up_apply_data();

    auto coo_mtx = gko::matrix::Coo<>::create(ref);
    auto dcoo_mtx = gko::matrix::Coo<>::create(cuda);

    x->convert_to(coo_mtx.get());
    dx->convert_to(dcoo_mtx.get());

    ASSERT_EQ(dcoo_mtx->get_num_stored_elements(),
              coo_mtx->get_num_stored_elements());
    GKO_ASSERT_MTX_NEAR(dcoo_mtx.get(), coo_mtx.get(), 1e-14);
}


TEST_F(Dense, ConvertToCsrIsEquivalentToRef)
{
    set_up_apply_data();

    auto csr_mtx = gko::matrix::Csr<>::create(ref);
    auto dcsr_mtx = gko::matrix::Csr<>::create(cuda);

    x->convert_to(csr_mtx.get());
    dx->convert_to(dcsr_mtx.get());

    GKO_ASSERT_MTX_NEAR(dcsr_mtx.get(), csr_mtx.get(), 1e-14);
}


TEST_F(Dense, ConvertToEllIsEquivalentToRef)
{
    set_up_apply_data();

    auto ell_mtx = gko::matrix::Ell<>::create(ref);
    auto dell_mtx = gko::matrix::Ell<>::create(cuda);

    x->convert_to(ell_mtx.get());
    dx->convert_to(dell_mtx.get());

    GKO_ASSERT_MTX_NEAR(dell_mtx.get(), ell_mtx.get(), 1e-14);
}


TEST_F(Dense, ConvertToSellpIsEquivalentToRef)
{
    set_up_apply_data();

    auto sellp_mtx = gko::matrix::Sellp<>::create(ref);
    auto dsellp_mtx = gko::matrix::Sellp<>::create(cuda);

    x->convert_to(sellp_mtx.get());
    dx->convert_to(dsellp_mtx.get());

    ASSERT_MTX_NEAR(sellp_mtx, dsellp_mtx, 1e-14);
}


TEST_F(Dense, CountNNZIsEquivalentToRef)
{
    set_up_apply_data();

    gko::size_type nnz;
    gko::size_type dnnz;

    gko::kernels::reference::dense::count_nonzeros(ref, x.get(), &nnz);
    gko::kernels::cuda::dense::count_nonzeros(cuda, dx.get(), &dnnz);

    ASSERT_EQ(nnz, dnnz);
}


TEST_F(Dense, CalculateNNZPerRowIsEquivalentToRef)
{
    set_up_apply_data();

    gko::Array<gko::size_type> nnz_per_row;
    nnz_per_row.set_executor(ref);
    nnz_per_row.resize_and_reset(x->get_size()[0]);

    gko::Array<gko::size_type> dnnz_per_row;
    dnnz_per_row.set_executor(cuda);
    dnnz_per_row.resize_and_reset(dx->get_size()[0]);

    gko::kernels::reference::dense::calculate_nonzeros_per_row(ref, x.get(),
                                                               &nnz_per_row);
    gko::kernels::cuda::dense::calculate_nonzeros_per_row(cuda, dx.get(),
                                                          &dnnz_per_row);

    auto tmp = gko::Array<gko::size_type>(ref, dnnz_per_row);
    for (auto i = 0; i < nnz_per_row.get_num_elems(); i++) {
        ASSERT_EQ(nnz_per_row.get_const_data()[i], tmp.get_const_data()[i]);
    }
}


TEST_F(Dense, CalculateMaxNNZPerRowIsEquivalentToRef)
{
    set_up_apply_data();

    gko::size_type max_nnz;
    gko::size_type dmax_nnz;

    gko::kernels::reference::dense::calculate_max_nnz_per_row(ref, x.get(),
                                                              &max_nnz);
    gko::kernels::cuda::dense::calculate_max_nnz_per_row(cuda, dx.get(),
                                                         &dmax_nnz);

    ASSERT_EQ(max_nnz, dmax_nnz);
}


TEST_F(Dense, CalculateTotalColsIsEquivalentToRef)
{
    set_up_apply_data();

    gko::size_type total_cols;
    gko::size_type dtotal_cols;

    gko::kernels::reference::dense::calculate_total_cols(
        ref, x.get(), &total_cols, 2, gko::matrix::default_slice_size);
    gko::kernels::cuda::dense::calculate_total_cols(
        cuda, dx.get(), &dtotal_cols, 2, gko::matrix::default_slice_size);

    ASSERT_EQ(total_cols, dtotal_cols);
}


}  // namespace
