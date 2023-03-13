/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include <ginkgo/core/matrix/batch_tridiagonal.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/range.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class BatchTridiagonal : public ::testing::Test {
protected:
    using value_type = T;
    using size_type = gko::size_type;

    /*
    BatchTridiagonal matrix:

    2  3  0  0
    4  1  5  0
    0  5  9  8
    0  0  8  4

    9  8  0  0  0
    4  3  5  0  0
    0  7  1  4  0
    0  0  8  2  1
    0  0  0  6  3

    */

    BatchTridiagonal() : exec(gko::ReferenceExecutor::create())
    {
        mtx = gko::matrix::BatchTridiagonal<value_type>::create(
            exec,
            std::vector<gko::dim<2>>{gko::dim<2>{4, 4}, gko::dim<2>{5, 5}});

        value_type* subdiag = mtx->get_sub_diagonal();
        value_type* maindiag = mtx->get_main_diagonal();
        value_type* superdiag = mtx->get_super_diagonal();

        //clang-format off
        subdiag[0] = 0.0;
        subdiag[1] = 4.0;
        subdiag[2] = 5.0;
        subdiag[3] = 8.0;
        subdiag[4] = 0.0;
        subdiag[5] = 4.0;
        subdiag[6] = 7.0;
        subdiag[7] = 8.0;
        subdiag[8] = 6.0;

        maindiag[0] = 2.0;
        maindiag[1] = 1.0;
        maindiag[2] = 9.0;
        maindiag[3] = 4.0;
        maindiag[4] = 9.0;
        maindiag[5] = 3.0;
        maindiag[6] = 1.0;
        maindiag[7] = 2.0;
        maindiag[8] = 3.0;

        superdiag[0] = 3.0;
        superdiag[1] = 5.0;
        superdiag[2] = 8.0;
        superdiag[3] = 0.0;
        superdiag[4] = 8.0;
        superdiag[5] = 5.0;
        superdiag[6] = 4.0;
        superdiag[7] = 1.0;
        superdiag[8] = 0.0;

        //clang-format on
    }


    static void assert_equal_to_original_mtx(
        gko::matrix::BatchTridiagonal<value_type>* m)
    {
        ASSERT_EQ(m->get_num_batch_entries(), 2);
        ASSERT_EQ(m->get_size().at(0), gko::dim<2>(4, 4));
        ASSERT_EQ(m->get_size().at(1), gko::dim<2>(5, 5));

        ASSERT_EQ(m->get_num_stored_elements(), (3 * 4) + (3 * 5));
        ASSERT_EQ(m->get_num_stored_elements_per_diagonal(0), 4);
        ASSERT_EQ(m->get_num_stored_elements_per_diagonal(1), 5);

        ASSERT_EQ(m->get_const_sub_diagonal(0)[0], value_type{0.0});
        EXPECT_EQ(m->get_const_sub_diagonal(0)[1], value_type{4.0});
        EXPECT_EQ(m->get_const_sub_diagonal(0)[2], value_type{5.0});
        EXPECT_EQ(m->get_const_sub_diagonal(0)[3], value_type{8.0});

        EXPECT_EQ(m->get_const_main_diagonal(0)[0], value_type{2.0});
        EXPECT_EQ(m->get_const_main_diagonal(0)[1], value_type{1.0});
        EXPECT_EQ(m->get_const_main_diagonal(0)[2], value_type{9.0});
        EXPECT_EQ(m->get_const_main_diagonal(0)[3], value_type{4.0});

        EXPECT_EQ(m->get_const_super_diagonal(0)[0], value_type{3.0});
        EXPECT_EQ(m->get_const_super_diagonal(0)[1], value_type{5.0});
        EXPECT_EQ(m->get_const_super_diagonal(0)[2], value_type{8.0});
        ASSERT_EQ(m->get_const_super_diagonal(0)[3], value_type{0.0});

        ASSERT_EQ(m->get_const_sub_diagonal(1)[0], value_type{0.0});
        EXPECT_EQ(m->get_const_sub_diagonal(1)[1], value_type{4.0});
        EXPECT_EQ(m->get_const_sub_diagonal(1)[2], value_type{7.0});
        EXPECT_EQ(m->get_const_sub_diagonal(1)[3], value_type{8.0});
        EXPECT_EQ(m->get_const_sub_diagonal(1)[4], value_type{6.0});

        EXPECT_EQ(m->get_const_main_diagonal(1)[0], value_type{9.0});
        EXPECT_EQ(m->get_const_main_diagonal(1)[1], value_type{3.0});
        EXPECT_EQ(m->get_const_main_diagonal(1)[2], value_type{1.0});
        EXPECT_EQ(m->get_const_main_diagonal(1)[3], value_type{2.0});
        EXPECT_EQ(m->get_const_main_diagonal(1)[4], value_type{3.0});

        EXPECT_EQ(m->get_const_super_diagonal(1)[0], value_type{8.0});
        EXPECT_EQ(m->get_const_super_diagonal(1)[1], value_type{5.0});
        EXPECT_EQ(m->get_const_super_diagonal(1)[2], value_type{4.0});
        EXPECT_EQ(m->get_const_super_diagonal(1)[3], value_type{1.0});
        ASSERT_EQ(m->get_const_super_diagonal(1)[4], value_type{0.0});
    }

    static void assert_empty(gko::matrix::BatchTridiagonal<value_type>* m)
    {
        ASSERT_EQ(m->get_num_batch_entries(), 0);
        ASSERT_EQ(m->get_num_stored_elements(), 0);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<gko::matrix::BatchTridiagonal<value_type>> mtx;
};

TYPED_TEST_SUITE(BatchTridiagonal, gko::test::ValueTypes);


TYPED_TEST(BatchTridiagonal, CanBeEmpty)
{
    auto empty = gko::matrix::BatchTridiagonal<TypeParam>::create(this->exec);
    this->assert_empty(empty.get());
}


TYPED_TEST(BatchTridiagonal, ReturnsNullValuesArrayWhenEmpty)
{
    auto empty = gko::matrix::BatchTridiagonal<TypeParam>::create(this->exec);
    ASSERT_EQ(empty->get_const_sub_diagonal(), nullptr);
    ASSERT_EQ(empty->get_const_main_diagonal(), nullptr);
    ASSERT_EQ(empty->get_const_super_diagonal(), nullptr);
}


TYPED_TEST(BatchTridiagonal, CanBeConstructedWithSize)
{
    using size_type = gko::size_type;
    auto m = gko::matrix::BatchTridiagonal<TypeParam>::create(
        this->exec,
        std::vector<gko::dim<2>>{gko::dim<2>{3, 3}, gko::dim<2>{4, 4}});

    ASSERT_EQ(m->get_num_batch_entries(), 2);
    ASSERT_EQ(m->get_size().at(0), gko::dim<2>(3, 3));
    ASSERT_EQ(m->get_size().at(1), gko::dim<2>(4, 4));
    ASSERT_EQ(m->get_num_stored_elements(), 21);
    ASSERT_EQ(m->get_num_stored_elements_per_diagonal(0), 3);
    ASSERT_EQ(m->get_num_stored_elements_per_diagonal(1), 4);
}


TYPED_TEST(BatchTridiagonal, CanBeConstructedFromExistingData)
{
    using value_type = typename TestFixture::value_type;
    using size_type = gko::size_type;

    // clang-format off
    value_type subdiag[] = {
       0.0, -1.0, //first mat sub-diagonal
       0.0, 3.0, 6.0 //second mat sub-diagonal
    };
    // clang-format on

    // clang-format off
    value_type maindiag[] = {

       1.0, 3.0, //first mat main-diagonal
       4.0, 5.0, -3.0 //second mat main-diagonal
      };
    // clang-format on

    // clang-format off
    value_type superdiag[] = {

       2.0, 0.0, //first mat super-diagonal
      -1.0, 1.0, 0.0 //second mat super-diagonal

      };
    // clang-format on

    auto m = gko::matrix::BatchTridiagonal<TypeParam>::create(
        this->exec,
        std::vector<gko::dim<2>>{gko::dim<2>{2, 2}, gko::dim<2>{3, 3}},
        gko::array<value_type>::view(this->exec, 5, subdiag),
        gko::array<value_type>::view(this->exec, 5, maindiag),
        gko::array<value_type>::view(this->exec, 5, superdiag));

    ASSERT_EQ(m->get_const_sub_diagonal(), subdiag);
    ASSERT_EQ(m->get_const_main_diagonal(), maindiag);
    ASSERT_EQ(m->get_const_super_diagonal(), superdiag);

    ASSERT_EQ(m->get_const_sub_diagonal()[0], value_type{0.0});
    ASSERT_EQ(m->get_const_sub_diagonal()[1], value_type{-1.0});
    ASSERT_EQ(m->get_const_sub_diagonal()[2], value_type{0.0});
    ASSERT_EQ(m->get_const_sub_diagonal()[3], value_type{3.0});
    ASSERT_EQ(m->get_const_sub_diagonal()[4], value_type{6.0});

    ASSERT_EQ(m->get_const_main_diagonal()[0], value_type{1.0});
    ASSERT_EQ(m->get_const_main_diagonal()[1], value_type{3.0});
    ASSERT_EQ(m->get_const_main_diagonal()[2], value_type{4.0});
    ASSERT_EQ(m->get_const_main_diagonal()[3], value_type{5.0});
    ASSERT_EQ(m->get_const_main_diagonal()[4], value_type{-3.0});

    ASSERT_EQ(m->get_const_super_diagonal()[0], value_type{2.0});
    ASSERT_EQ(m->get_const_super_diagonal()[1], value_type{0.0});
    ASSERT_EQ(m->get_const_super_diagonal()[2], value_type{-1.0});
    ASSERT_EQ(m->get_const_super_diagonal()[3], value_type{1.0});
    ASSERT_EQ(m->get_const_super_diagonal()[4], value_type{0.0});
}


TYPED_TEST(BatchTridiagonal, CanBeConstructedFromExistingConstData)
{
    using value_type = typename TestFixture::value_type;
    using size_type = gko::size_type;

    // clang-format off
    value_type subdiag[] = {
       0.0, -1.0, //first mat sub-diagonal
       0.0, 3.0, 6.0 //second mat sub-diagonal
    };
    // clang-format on

    // clang-format off
    value_type maindiag[] = {

       1.0, 3.0, //first mat main-diagonal
       4.0, 5.0, -3.0 //second mat main-diagonal
      };
    // clang-format on

    // clang-format off
    value_type superdiag[] = {

       2.0, 0.0, //first mat super-diagonal
      -1.0, 1.0, 0.0 //second mat super-diagonal

      };
    // clang-format on

    auto m = gko::matrix::BatchTridiagonal<TypeParam>::create_const(
        this->exec,
        std::vector<gko::dim<2>>{gko::dim<2>{2, 2}, gko::dim<2>{3, 3}},
        gko::array<value_type>::const_view(this->exec, 5, subdiag),
        gko::array<value_type>::const_view(this->exec, 5, maindiag),
        gko::array<value_type>::const_view(this->exec, 5, superdiag));

    ASSERT_EQ(m->get_const_sub_diagonal(), subdiag);
    ASSERT_EQ(m->get_const_main_diagonal(), maindiag);
    ASSERT_EQ(m->get_const_super_diagonal(), superdiag);

    ASSERT_EQ(m->get_const_sub_diagonal()[0], value_type{0.0});
    ASSERT_EQ(m->get_const_sub_diagonal()[1], value_type{-1.0});
    ASSERT_EQ(m->get_const_sub_diagonal()[2], value_type{0.0});
    ASSERT_EQ(m->get_const_sub_diagonal()[3], value_type{3.0});
    ASSERT_EQ(m->get_const_sub_diagonal()[4], value_type{6.0});

    ASSERT_EQ(m->get_const_main_diagonal()[0], value_type{1.0});
    ASSERT_EQ(m->get_const_main_diagonal()[1], value_type{3.0});
    ASSERT_EQ(m->get_const_main_diagonal()[2], value_type{4.0});
    ASSERT_EQ(m->get_const_main_diagonal()[3], value_type{5.0});
    ASSERT_EQ(m->get_const_main_diagonal()[4], value_type{-3.0});

    ASSERT_EQ(m->get_const_super_diagonal()[0], value_type{2.0});
    ASSERT_EQ(m->get_const_super_diagonal()[1], value_type{0.0});
    ASSERT_EQ(m->get_const_super_diagonal()[2], value_type{-1.0});
    ASSERT_EQ(m->get_const_super_diagonal()[3], value_type{1.0});
    ASSERT_EQ(m->get_const_super_diagonal()[4], value_type{0.0});
}


TYPED_TEST(BatchTridiagonal,
           CanBeConstructedFromBatchTridiagonalMatricesByDuplication)
{
    using value_type = typename TestFixture::value_type;
    using size_type = gko::size_type;

    // clang-format off
    value_type subdiag[] = {

        0.0, 1.0, 5.0,  // sub-diagonal
        0.0, 3.0, 6.0   // sub-diagonal
    };

    value_type maindiag[] = {

        1.0, 7.0, -7.0,  // main-diagonal
        4.0, 5.0, -3.0   // main-diagonal
    };

    value_type superdiag[] = {

        -1.0, 3.0, 0.0,  // super-diagonal
        -1.0, 1.0, 0.0   // super-diagonal

    };
    // clang-format on

    auto m = gko::matrix::BatchTridiagonal<TypeParam>::create(
        this->exec, gko::batch_dim<2>{2, gko::dim<2>{3, 3}},
        gko::array<value_type>::view(this->exec, 6, subdiag),
        gko::array<value_type>::view(this->exec, 6, maindiag),
        gko::array<value_type>::view(this->exec, 6, superdiag));

    auto bat_m_created_by_dupl =
        gko::matrix::BatchTridiagonal<TypeParam>::create(this->exec, 2,
                                                         m.get());
    // clang-format off
    value_type subdiag_new[] = {

       0.0, 1.0, 5.0, //sub-diagonal
       0.0, 3.0, 6.0, //sub-diagonal
       0.0, 1.0, 5.0, //sub-diagonal
       0.0, 3.0, 6.0 //sub-diagonal

    };

    value_type maindiag_new[] = {

       1.0, 7.0, -7.0, //main-diagonal
       4.0, 5.0, -3.0, //main-diagonal
       1.0, 7.0, -7.0, //main-diagonal
       4.0, 5.0, -3.0 //main-diagonal

    };

    value_type superdiag_new[] = {

      -1.0, 3.0, 0.0, //super-diagonal
      -1.0, 1.0, 0.0, //super-diagonal
      -1.0, 3.0, 0.0, //super-diagonal
      -1.0, 1.0, 0.0 //super-diagonal

    };
    // clang-format on

    auto m_new = gko::matrix::BatchTridiagonal<TypeParam>::create(
        this->exec, gko::batch_dim<2>(4, gko::dim<2>{3, 3}),
        gko::array<value_type>::view(this->exec, 12, subdiag_new),
        gko::array<value_type>::view(this->exec, 12, maindiag_new),
        gko::array<value_type>::view(this->exec, 12, superdiag_new));

    GKO_ASSERT_BATCH_MTX_NEAR(bat_m_created_by_dupl.get(), m_new.get(), 1e-14);
}


TYPED_TEST(BatchTridiagonal, KnowsItsSizeAndValues)
{
    this->assert_equal_to_original_mtx(this->mtx.get());
}


TYPED_TEST(BatchTridiagonal, CanBeCopied)
{
    auto mtx_copy =
        gko::matrix::BatchTridiagonal<TypeParam>::create(this->exec);
    mtx_copy->copy_from(this->mtx.get());
    this->assert_equal_to_original_mtx(this->mtx.get());
    this->assert_equal_to_original_mtx(mtx_copy.get());
}


TYPED_TEST(BatchTridiagonal, CanBeMoved)
{
    auto mtx_copy =
        gko::matrix::BatchTridiagonal<TypeParam>::create(this->exec);
    mtx_copy->copy_from(std::move(this->mtx));
    this->assert_equal_to_original_mtx(mtx_copy.get());
}


TYPED_TEST(BatchTridiagonal, CanBeCloned)
{
    auto mtx_clone = this->mtx->clone();
    this->assert_equal_to_original_mtx(
        dynamic_cast<decltype(this->mtx.get())>(mtx_clone.get()));
}


TYPED_TEST(BatchTridiagonal, CanBeCleared)
{
    this->mtx->clear();
    this->assert_empty(this->mtx.get());
}


TYPED_TEST(BatchTridiagonal, CanBeReadFromMatrixData)
{
    using value_type = typename TestFixture::value_type;
    auto m = gko::matrix::BatchTridiagonal<TypeParam>::create(this->exec);

    /*

        first matrix:
        2  4  0
        3  6  1
        0  0  9

        second matrix:
        4  3
        3  7

    */

    // clang-format off
    m->read({gko::matrix_data<TypeParam>{{3, 3},
                                         {{0, 0, 2.0},
                                          {0, 1, 4.0},
                                          {1, 0, 3.0},
                                          {1, 1, 6.0},
                                          {1, 2, 1.0},
                                          {2, 2, 9.0}}},
             gko::matrix_data<TypeParam>{{2, 2},
                                         {{0, 0, 4.0},
                                          {0, 1, 3.0},
                                          {1, 0, 3.0},
                                          {1, 1, 7.0}}}});
    // clang-format on

    ASSERT_EQ(m->get_size().at(0), gko::dim<2>(3, 3));
    ASSERT_EQ(m->get_size().at(1), gko::dim<2>(2, 2));
    ASSERT_EQ(m->get_num_stored_elements(), 15);
    ASSERT_EQ(m->get_num_stored_elements_per_diagonal(0), 3);
    ASSERT_EQ(m->get_num_stored_elements_per_diagonal(1), 2);

    ASSERT_EQ(m->get_const_sub_diagonal(0)[0], value_type{0.0});
    EXPECT_EQ(m->get_const_sub_diagonal(0)[1], value_type{3.0});
    EXPECT_EQ(m->get_const_sub_diagonal(0)[2], value_type{0.0});
    ASSERT_EQ(m->get_const_sub_diagonal(1)[0], value_type{0.0});
    EXPECT_EQ(m->get_const_sub_diagonal(1)[1], value_type{3.0});

    EXPECT_EQ(m->get_const_main_diagonal(0)[0], value_type{2.0});
    EXPECT_EQ(m->get_const_main_diagonal(0)[1], value_type{6.0});
    EXPECT_EQ(m->get_const_main_diagonal(0)[2], value_type{9.0});
    EXPECT_EQ(m->get_const_main_diagonal(1)[0], value_type{4.0});
    EXPECT_EQ(m->get_const_main_diagonal(1)[1], value_type{7.0});

    EXPECT_EQ(m->get_const_super_diagonal(0)[0], value_type{4.0});
    EXPECT_EQ(m->get_const_super_diagonal(0)[1], value_type{1.0});
    ASSERT_EQ(m->get_const_super_diagonal(0)[2], value_type{0.0});
    EXPECT_EQ(m->get_const_super_diagonal(1)[0], value_type{3.0});
    ASSERT_EQ(m->get_const_super_diagonal(1)[1], value_type{0.0});
}


TYPED_TEST(BatchTridiagonal, CanBeReadFromMatrixAssemblyData)
{
    /*

        first matrix:
        2  4  0
        3  6  1
        0  0  9

        second matrix:
        4  3
        3  7

    */
    using value_type = typename TestFixture::value_type;
    auto m = gko::matrix::BatchTridiagonal<TypeParam>::create(this->exec);
    gko::matrix_assembly_data<TypeParam> data1(gko::dim<2>{3, 3});
    data1.set_value(0, 0, 2.0);
    data1.set_value(0, 1, 4.0);
    data1.set_value(1, 0, 3.0);
    data1.set_value(1, 1, 6.0);
    data1.set_value(1, 2, 1.0);
    data1.set_value(2, 2, 9.0);
    gko::matrix_assembly_data<TypeParam> data2(gko::dim<2>{2, 2});
    data2.set_value(0, 0, 4.0);
    data2.set_value(0, 1, 3.0);
    data2.set_value(1, 0, 3.0);
    data2.set_value(1, 1, 7.0);
    auto data = std::vector<gko::matrix_assembly_data<TypeParam>>{data1, data2};

    m->read(data);


    ASSERT_EQ(m->get_size().at(0), gko::dim<2>(3, 3));
    ASSERT_EQ(m->get_size().at(1), gko::dim<2>(2, 2));
    ASSERT_EQ(m->get_num_stored_elements(), 15);
    ASSERT_EQ(m->get_num_stored_elements_per_diagonal(0), 3);
    ASSERT_EQ(m->get_num_stored_elements_per_diagonal(1), 2);

    ASSERT_EQ(m->get_const_sub_diagonal(0)[0], value_type{0.0});
    EXPECT_EQ(m->get_const_sub_diagonal(0)[1], value_type{3.0});
    EXPECT_EQ(m->get_const_sub_diagonal(0)[2], value_type{0.0});
    ASSERT_EQ(m->get_const_sub_diagonal(1)[0], value_type{0.0});
    EXPECT_EQ(m->get_const_sub_diagonal(1)[1], value_type{3.0});

    EXPECT_EQ(m->get_const_main_diagonal(0)[0], value_type{2.0});
    EXPECT_EQ(m->get_const_main_diagonal(0)[1], value_type{6.0});
    EXPECT_EQ(m->get_const_main_diagonal(0)[2], value_type{9.0});
    EXPECT_EQ(m->get_const_main_diagonal(1)[0], value_type{4.0});
    EXPECT_EQ(m->get_const_main_diagonal(1)[1], value_type{7.0});

    EXPECT_EQ(m->get_const_super_diagonal(0)[0], value_type{4.0});
    EXPECT_EQ(m->get_const_super_diagonal(0)[1], value_type{1.0});
    ASSERT_EQ(m->get_const_super_diagonal(0)[2], value_type{0.0});
    EXPECT_EQ(m->get_const_super_diagonal(1)[0], value_type{3.0});
    ASSERT_EQ(m->get_const_super_diagonal(1)[1], value_type{0.0});
}


TYPED_TEST(BatchTridiagonal, GeneratesCorrectMatrixData)
{
    using value_type = typename TestFixture::value_type;
    using tpl = typename gko::matrix_data<TypeParam>::nonzero_type;
    std::vector<gko::matrix_data<TypeParam>> data;

    this->mtx->write(data);

    ASSERT_EQ(data[0].size, gko::dim<2>(4, 4));

    ASSERT_EQ(data[0].nonzeros.size(), 10);
    EXPECT_EQ(data[0].nonzeros[0], tpl(0, 0, value_type{2.0}));
    EXPECT_EQ(data[0].nonzeros[1], tpl(0, 1, value_type{3.0}));
    EXPECT_EQ(data[0].nonzeros[2], tpl(1, 0, value_type{4.0}));
    EXPECT_EQ(data[0].nonzeros[3], tpl(1, 1, value_type{1.0}));
    EXPECT_EQ(data[0].nonzeros[4], tpl(1, 2, value_type{5.0}));
    EXPECT_EQ(data[0].nonzeros[5], tpl(2, 1, value_type{5.0}));
    EXPECT_EQ(data[0].nonzeros[6], tpl(2, 2, value_type{9.0}));
    EXPECT_EQ(data[0].nonzeros[7], tpl(2, 3, value_type{8.0}));
    EXPECT_EQ(data[0].nonzeros[8], tpl(3, 2, value_type{8.0}));
    EXPECT_EQ(data[0].nonzeros[9], tpl(3, 3, value_type{4.0}));
    ASSERT_EQ(data[1].size, gko::dim<2>(5, 5));
    ASSERT_EQ(data[1].nonzeros.size(), 13);
    EXPECT_EQ(data[1].nonzeros[0], tpl(0, 0, value_type{9.0}));
    EXPECT_EQ(data[1].nonzeros[1], tpl(0, 1, value_type{8.0}));
    EXPECT_EQ(data[1].nonzeros[2], tpl(1, 0, value_type{4.0}));
    EXPECT_EQ(data[1].nonzeros[3], tpl(1, 1, value_type{3.0}));
    EXPECT_EQ(data[1].nonzeros[4], tpl(1, 2, value_type{5.0}));
    EXPECT_EQ(data[1].nonzeros[5], tpl(2, 1, value_type{7.0}));
    EXPECT_EQ(data[1].nonzeros[6], tpl(2, 2, value_type{1.0}));
    EXPECT_EQ(data[1].nonzeros[7], tpl(2, 3, value_type{4.0}));
    EXPECT_EQ(data[1].nonzeros[8], tpl(3, 2, value_type{8.0}));
    EXPECT_EQ(data[1].nonzeros[9], tpl(3, 3, value_type{2.0}));
    EXPECT_EQ(data[1].nonzeros[10], tpl(3, 4, value_type{1.0}));
    EXPECT_EQ(data[1].nonzeros[11], tpl(4, 3, value_type{6.0}));
    EXPECT_EQ(data[1].nonzeros[12], tpl(4, 4, value_type{3.0}));
}


}  // namespace
