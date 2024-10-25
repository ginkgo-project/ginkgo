// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/preconditioner/gauss_seidel.hpp>
#include <ginkgo/core/preconditioner/sor.hpp>
#include <ginkgo/core/solver/triangular.hpp>

#include "core/test/utils.hpp"
#include "core/utils/matrix_utils.hpp"


template <typename ValueIndexType>
class GaussSeidel : public ::testing::Test {
public:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using csr_type = gko::matrix::Csr<value_type, index_type>;
    using gs_type = gko::preconditioner::GaussSeidel<value_type, index_type>;
    using sor_type = gko::preconditioner::Sor<value_type, index_type>;
    using ltrs_type = gko::solver::LowerTrs<value_type, index_type>;
    using utrs_type = gko::solver::UpperTrs<value_type, index_type>;

    GaussSeidel()
    {
        auto data =
            gko::test::generate_random_matrix_data<value_type, index_type>(
                10, 10, std::uniform_int_distribution<>(2, 6),
                std::uniform_real_distribution<>(1, 2), engine);
        gko::utils::make_symmetric(data);
        gko::utils::make_unit_diagonal(data);
        mtx->read(data);
    }

    std::default_random_engine engine;
    std::shared_ptr<gko::ReferenceExecutor> exec =
        gko::ReferenceExecutor::create();
    std::shared_ptr<csr_type> mtx = csr_type::create(exec);
};

TYPED_TEST_SUITE(GaussSeidel, gko::test::ValueIndexTypesWithHalf,
                 PairTypenameNameGenerator);


TYPED_TEST(GaussSeidel, GenerateSameAsSor)
{
    using real_type = gko::remove_complex<typename TestFixture::value_type>;
    using gs_type = typename TestFixture::gs_type;
    using sor_type = typename TestFixture::sor_type;
    using composition_type = typename sor_type::composition_type;
    using csr_type = typename TestFixture::csr_type;
    using ltrs_type = typename TestFixture::ltrs_type;

    auto gs = gs_type ::build().on(this->exec)->generate(this->mtx);
    auto sor = sor_type::build()
                   .with_relaxation_factor(real_type{1.0})
                   .on(this->exec)
                   ->generate(this->mtx);

    auto gs_comp = dynamic_cast<composition_type*>(gs.get());
    auto sor_comp = dynamic_cast<composition_type*>(sor.get());
    ASSERT_TRUE(gs_comp);
    ASSERT_TRUE(sor_comp);
    ASSERT_EQ(gs_comp->get_operators().size(),
              sor_comp->get_operators().size());
    GKO_ASSERT_MTX_NEAR(
        dynamic_cast<const ltrs_type*>(gs_comp->get_operators()[0].get())
            ->get_system_matrix(),
        dynamic_cast<const ltrs_type*>(sor_comp->get_operators()[0].get())
            ->get_system_matrix(),
        0.0);
}

TYPED_TEST(GaussSeidel, GenerateSymmetricSameAsSor)
{
    using real_type = gko::remove_complex<typename TestFixture::value_type>;
    using gs_type = typename TestFixture::gs_type;
    using sor_type = typename TestFixture::sor_type;
    using composition_type = typename sor_type::composition_type;
    using ltrs_type = typename TestFixture::ltrs_type;
    using utrs_type = typename TestFixture::utrs_type;

    auto gs = gs_type ::build()
                  .with_symmetric(true)
                  .on(this->exec)
                  ->generate(this->mtx);
    auto sor = sor_type::build()
                   .with_symmetric(true)
                   .with_relaxation_factor(real_type{1.0})
                   .on(this->exec)
                   ->generate(this->mtx);

    auto gs_comp = dynamic_cast<composition_type*>(gs.get());
    auto sor_comp = dynamic_cast<composition_type*>(sor.get());
    ASSERT_TRUE(gs_comp);
    ASSERT_TRUE(sor_comp);
    ASSERT_EQ(gs_comp->get_operators().size(),
              sor_comp->get_operators().size());
    GKO_ASSERT_MTX_NEAR(
        dynamic_cast<const utrs_type*>(gs_comp->get_operators()[0].get())
            ->get_system_matrix(),
        dynamic_cast<const utrs_type*>(sor_comp->get_operators()[0].get())
            ->get_system_matrix(),
        0.0);
    GKO_ASSERT_MTX_NEAR(
        dynamic_cast<const ltrs_type*>(gs_comp->get_operators()[1].get())
            ->get_system_matrix(),
        dynamic_cast<const ltrs_type*>(sor_comp->get_operators()[1].get())
            ->get_system_matrix(),
        0.0);
}
