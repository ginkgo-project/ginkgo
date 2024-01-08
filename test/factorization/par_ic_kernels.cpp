// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/par_ic_kernels.hpp"


#include <algorithm>
#include <fstream>
#include <memory>
#include <random>
#include <string>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/factorization/factorization_kernels.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/test/utils.hpp"
#include "matrices/config.hpp"
#include "test/utils/executor.hpp"


template <typename ValueIndexType>
class ParIc : public CommonTestFixture {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Coo = gko::matrix::Coo<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;

    ParIc() : mtx_size(624, 624), rand_engine(43456)
    {
        mtx_l = gko::test::generate_random_lower_triangular_matrix<Csr>(
            mtx_size[0], false,
            std::uniform_int_distribution<index_type>(10, mtx_size[0]),
            std::normal_distribution<gko::remove_complex<value_type>>(0, 10.0),
            rand_engine, ref);
        dmtx_ani = Csr::create(exec);
        dmtx_l_ani = Csr::create(exec);
        dmtx_l_ani_init = Csr::create(exec);
        dmtx_l = gko::clone(exec, mtx_l);
        std::string file_name(gko::matrices::location_ani4_mtx);
        auto input_file = std::ifstream(file_name, std::ios::in);
        mtx_ani = gko::read<Csr>(input_file, ref);
        mtx_ani->sort_by_column_index();

        {
            mtx_l_ani = Csr::create(ref, mtx_ani->get_size());
            gko::matrix::CsrBuilder<value_type, index_type> l_builder(
                mtx_l_ani);
            gko::kernels::reference::factorization::initialize_row_ptrs_l(
                ref, mtx_ani.get(), mtx_l_ani->get_row_ptrs());
            auto l_nnz =
                mtx_l_ani->get_const_row_ptrs()[mtx_ani->get_size()[0]];
            l_builder.get_col_idx_array().resize_and_reset(l_nnz);
            l_builder.get_value_array().resize_and_reset(l_nnz);
            gko::kernels::reference::factorization::initialize_l(
                ref, mtx_ani.get(), mtx_l_ani.get(), false);
            mtx_l_ani_init = gko::clone(ref, mtx_l_ani);
            gko::kernels::reference::par_ic_factorization::init_factor(
                ref, mtx_l_ani_init.get());
        }
        dmtx_ani->copy_from(mtx_ani);
        dmtx_l_ani->copy_from(mtx_l_ani);
        dmtx_l_ani_init->copy_from(mtx_l_ani_init);
    }

    gko::dim<2> mtx_size;
    std::default_random_engine rand_engine;

    std::unique_ptr<Csr> mtx_l;
    std::unique_ptr<Csr> mtx_ani;
    std::unique_ptr<Csr> mtx_l_ani;
    std::unique_ptr<Csr> mtx_l_ani_init;

    std::unique_ptr<Csr> dmtx_l;
    std::unique_ptr<Csr> dmtx_ani;
    std::unique_ptr<Csr> dmtx_l_ani;
    std::unique_ptr<Csr> dmtx_l_ani_init;
};

TYPED_TEST_SUITE(ParIc, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(ParIc, KernelInitFactorIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;

    gko::kernels::reference::par_ic_factorization::init_factor(
        this->ref, this->mtx_l.get());
    gko::kernels::EXEC_NAMESPACE::par_ic_factorization::init_factor(
        this->exec, this->dmtx_l.get());

    GKO_ASSERT_MTX_NEAR(this->mtx_l, this->dmtx_l, r<value_type>::value);
}


TYPED_TEST(ParIc, KernelComputeFactorIsEquivalentToRef)
{
    using Csr = typename TestFixture::Csr;
    using Coo = typename TestFixture::Coo;
    auto square_size = this->mtx_ani->get_size();
    auto mtx_l_coo = Coo::create(this->ref, square_size);
    this->mtx_l_ani->convert_to(mtx_l_coo);
    auto dmtx_l_coo = gko::clone(this->exec, mtx_l_coo);

    gko::kernels::reference::par_ic_factorization::compute_factor(
        this->ref, 1, mtx_l_coo.get(), this->mtx_l_ani_init.get());
    gko::kernels::EXEC_NAMESPACE::par_ic_factorization::compute_factor(
        this->exec, 100, dmtx_l_coo.get(), this->dmtx_l_ani_init.get());

    GKO_ASSERT_MTX_NEAR(this->mtx_l_ani_init, this->dmtx_l_ani_init, 1e-4);
}
