/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include <ginkgo/core/multigrid/amgx_pgm.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class AmgxPgm : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using RestrictProlong = gko::multigrid::AmgxPgm<value_type, index_type>;
    using T = value_type;
    AmgxPgm()
        : exec(gko::ReferenceExecutor::create()),
          amgxpgm_factory(RestrictProlong::build()
                              .with_max_iterations(2u)
                              .with_max_unassigned_percentage(0.1)
                              .on(exec)),
          mtx(Mtx::create(exec, gko::dim<2>(5, 5), 15,
                          std::make_shared<typename Mtx::classical>())),
          agg(exec, 5),
          coarse(Mtx::create(exec, gko::dim<2>(2, 2), 4,
                             std::make_shared<typename Mtx::classical>()))

    {
        this->create_mtx(mtx.get(), &agg, coarse.get());
        rstr_prlg = amgxpgm_factory->generate(mtx);
    }

    void create_mtx(Mtx *fine, gko::Array<index_type> *agg, Mtx *coarse)
    {
        auto f_vals = fine->get_values();
        auto f_cols = fine->get_col_idxs();
        auto f_rows = fine->get_row_ptrs();
        auto c_vals = coarse->get_values();
        auto c_cols = coarse->get_col_idxs();
        auto c_rows = coarse->get_row_ptrs();
        auto agg_val = agg->get_data();
        /* this matrix is stored:
         *  5 -3 -3  0  0
         * -3  5  0 -2 -1
         * -3  0  5  0 -1
         *  0 -3  0  5  0
         *  0 -2 -2  0  5
         */
        f_vals[0] = 5;
        f_vals[1] = -3;
        f_vals[2] = -3;
        f_vals[3] = -3;
        f_vals[4] = 5;
        f_vals[5] = -2;
        f_vals[6] = -1;
        f_vals[7] = -3;
        f_vals[8] = 5;
        f_vals[9] = -1;
        f_vals[10] = -3;
        f_vals[11] = 5;
        f_vals[12] = -2;
        f_vals[13] = -2;
        f_vals[14] = 5;

        f_rows[0] = 0;
        f_rows[1] = 3;
        f_rows[2] = 7;
        f_rows[3] = 10;
        f_rows[4] = 12;
        f_rows[5] = 15;

        f_cols[0] = 0;
        f_cols[1] = 1;
        f_cols[2] = 2;
        f_cols[3] = 0;
        f_cols[4] = 1;
        f_cols[5] = 3;
        f_cols[6] = 4;
        f_cols[7] = 0;
        f_cols[8] = 2;
        f_cols[9] = 4;
        f_cols[10] = 1;
        f_cols[11] = 3;
        f_cols[12] = 1;
        f_cols[13] = 2;
        f_cols[14] = 4;

        agg_val[0] = 0;
        agg_val[1] = 1;
        agg_val[2] = 0;
        agg_val[3] = 1;
        agg_val[4] = 0;
        /* this coarse is stored:
         *  6 -5
         * -4  5
         */
        c_vals[0] = 6;
        c_vals[1] = -5;
        c_vals[2] = -4;
        c_vals[3] = 5;

        c_rows[0] = 0;
        c_rows[1] = 2;
        c_rows[2] = 4;

        c_cols[0] = 0;
        c_cols[1] = 1;
        c_cols[2] = 0;
        c_cols[3] = 1;
    }

    static void assert_same_matrices(const Mtx *m1, const Mtx *m2)
    {
        ASSERT_EQ(m1->get_size()[0], m2->get_size()[0]);
        ASSERT_EQ(m1->get_size()[1], m2->get_size()[1]);
        ASSERT_EQ(m1->get_num_stored_elements(), m2->get_num_stored_elements());
        for (gko::size_type i = 0; i < m1->get_size() + 1; i++) {
            ASSERT_EQ(m1->get_const_row_ptrs()[i], m2->get_const_row_ptrs()[i]);
        }
        for (gko::size_type i = 0; i < m1->get_num_stored_elements(); ++i) {
            EXPECT_EQ(m1->get_const_values()[i], m2->get_const_values()[i]);
            EXPECT_EQ(m1->get_const_col_idxs()[i], m2->get_const_col_idxs()[i]);
        }
    }

    static void assert_same_agg(const index_type *m1, const index_type *m2,
                                gko::size_type len)
    {
        for (gko::size_type i = 0; i < len; ++i) {
            EXPECT_EQ(m1[i], m2[i]);
        }
    }

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<Mtx> coarse;
    gko::Array<index_type> agg;
    std::unique_ptr<typename RestrictProlong::Factory> amgxpgm_factory;
    std::unique_ptr<RestrictProlong> rstr_prlg;
};

TYPED_TEST_CASE(AmgxPgm, gko::test::ValueIndexTypes);


TYPED_TEST(AmgxPgm, FactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->amgxpgm_factory->get_executor(), this->exec);
}


TYPED_TEST(AmgxPgm, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using RestrictProlong = typename TestFixture::RestrictProlong;
    auto copy = this->amgxpgm_factory->generate(Mtx::create(this->exec));

    // copy->copy_from(this->rstr_prlg.get());

    // auto copy_mtx =
    //     static_cast<RestrictProlong *>(copy.get())->get_system_matrix();
    // auto copy_agg = static_cast<RestrictProlong
    // *>(copy.get())->get_const_agg(); auto copy_coarse =
    // copy->get_coarse_operator();

    // this->assert_same_matrices(static_cast<const Mtx *>(copy_mtx.get()),
    //                            this->mtx.get());
    // this->assert_same_agg(copy_agg, this->agg.get_data(),
    // this->agg.get_num_elems()); this->assert_same_matrices(static_cast<const
    // Mtx *>(copy_coarse.get()), this->coarse.get());
}


TYPED_TEST(AmgxPgm, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using RestrictProlong = typename TestFixture::RestrictProlong;
    auto copy = this->amgxpgm_factory->generate(Mtx::create(this->exec));

    copy->copy_from(std::move(this->rstr_prlg));

    auto copy_mtx =
        static_cast<RestrictProlong *>(copy.get())->get_system_matrix();
    auto copy_agg = static_cast<RestrictProlong *>(copy.get())->get_const_agg();
    auto copy_coarse = copy->get_coarse_operator();

    this->assert_same_matrices(static_cast<const Mtx *>(copy_mtx.get()),
                               this->mtx.get());
    this->assert_same_agg(copy_agg, this->agg.get_data(),
                          this->agg.get_num_elems());
    this->assert_same_matrices(static_cast<const Mtx *>(copy_coarse.get()),
                               this->coarse.get());
}


TYPED_TEST(AmgxPgm, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    using RestrictProlong = typename TestFixture::RestrictProlong;
    auto copy = this->amgxpgm_factory->generate(Mtx::create(this->exec));

    auto clone = this->rstr_prlg->clone();

    auto clone_mtx =
        static_cast<RestrictProlong *>(clone.get())->get_system_matrix();
    auto clone_agg =
        static_cast<RestrictProlong *>(clone.get())->get_const_agg();
    auto clone_coarse = clone->get_coarse_operator();

    this->assert_same_matrices(static_cast<const Mtx *>(clone_mtx.get()),
                               this->mtx.get());
    this->assert_same_agg(clone_agg, this->agg.get_data(),
                          this->agg.get_num_elems());
    this->assert_same_matrices(static_cast<const Mtx *>(clone_coarse.get()),
                               this->coarse.get());
}


TYPED_TEST(AmgxPgm, CanBeCleared)
{
    using RestrictProlong = typename TestFixture::RestrictProlong;
    this->rstr_prlg->clear();

    auto mtx = static_cast<RestrictProlong *>(this->rstr_prlg.get())
                   ->get_system_matrix();
    auto coarse = this->rstr_prlg->get_coarse_operator();
    auto agg = static_cast<RestrictProlong *>(this->rstr_prlg.get())->get_agg();
    ASSERT_EQ(mtx, nullptr);
    ASSERT_EQ(coarse, nullptr);
    ASSERT_EQ(agg, nullptr);
}


}  // namespace
