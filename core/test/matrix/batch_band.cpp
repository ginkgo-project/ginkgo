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

#include <ginkgo/core/matrix/batch_band.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/range.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class BatchBand : public ::testing::Test {
protected:
    using value_type = T;
    using size_type = gko::size_type;

    /*
    BatchBand matrix:

    2  3  11  0
    4  1  7   13
    0  0  9   8
    0  0  8   4

    9   8    0   0   0
    0   3    5   0   0
    12  7    1   4   0
    2   15   8   2   1
    0   4   19   6   3

    */

    BatchBand() : exec(gko::ReferenceExecutor::create())
    {
        mtx = gko::matrix::BatchBand<value_type>::create(
            exec,
            std::vector<gko::dim<2>>{gko::dim<2>{4, 4}, gko::dim<2>{5, 5}},
            std::vector<gko::size_type>{1, 3},
            std::vector<gko::size_type>{2, 1});

        //clang-format off
        mtx->at_in_reference_to_dense_layout(0, 0, 0) = 2.0;
        mtx->at_in_reference_to_dense_layout(0, 1, 0) = 4.0;
        mtx->at_in_reference_to_dense_layout(0, 0, 1) = 3.0;
        mtx->at_in_reference_to_dense_layout(0, 1, 1) = 1.0;
        mtx->at_in_reference_to_dense_layout(0, 2, 1) = 0.0;
        mtx->at_in_reference_to_dense_layout(0, 0, 2) = 11.0;
        mtx->at_in_reference_to_dense_layout(0, 1, 2) = 7.0;
        mtx->at_in_reference_to_dense_layout(0, 2, 2) = 9.0;
        mtx->at_in_reference_to_dense_layout(0, 3, 2) = 8.0;
        mtx->at_in_reference_to_dense_layout(0, 1, 3) = 13.0;
        mtx->at_in_reference_to_dense_layout(0, 2, 3) = 8.0;
        mtx->at_in_reference_to_dense_layout(0, 3, 3) = 4.0;

        mtx->at_in_reference_to_dense_layout(1, 0, 0) = 9.0;
        mtx->at_in_reference_to_dense_layout(1, 1, 0) = 0.0;
        mtx->at_in_reference_to_dense_layout(1, 2, 0) = 12.0;
        mtx->at_in_reference_to_dense_layout(1, 3, 0) = 2.0;
        mtx->at_in_reference_to_dense_layout(1, 0, 1) = 8.0;
        mtx->at_in_reference_to_dense_layout(1, 1, 1) = 3.0;
        mtx->at_in_reference_to_dense_layout(1, 2, 1) = 7.0;
        mtx->at_in_reference_to_dense_layout(1, 3, 1) = 15.0;
        mtx->at_in_reference_to_dense_layout(1, 4, 1) = 4.0;
        mtx->at_in_reference_to_dense_layout(1, 1, 2) = 5.0;
        mtx->at_in_reference_to_dense_layout(1, 2, 2) = 1.0;
        mtx->at_in_reference_to_dense_layout(1, 3, 2) = 8.0;
        mtx->at_in_reference_to_dense_layout(1, 4, 2) = 19.0;
        mtx->at_in_reference_to_dense_layout(1, 2, 3) = 4.0;
        mtx->at_in_reference_to_dense_layout(1, 3, 3) = 2.0;
        mtx->at_in_reference_to_dense_layout(1, 4, 3) = 6.0;
        mtx->at_in_reference_to_dense_layout(1, 3, 4) = 1.0;
        mtx->at_in_reference_to_dense_layout(1, 4, 4) = 3.0;

        //clang-format on
    }


    static void assert_equal_to_original_mtx(
        gko::matrix::BatchBand<value_type>* m)
    {
        ASSERT_EQ(m->get_num_batch_entries(), 2);

        ASSERT_EQ(m->get_size().at(0), gko::dim<2>(4, 4));
        ASSERT_EQ(m->get_size().at(1), gko::dim<2>(5, 5));

        ASSERT_EQ(m->get_num_subdiagonals().at(0), 1);
        ASSERT_EQ(m->get_num_subdiagonals().at(1), 3);
        ASSERT_EQ(m->get_num_superdiagonals().at(0), 2);
        ASSERT_EQ(m->get_num_superdiagonals().at(1), 1);

        ASSERT_EQ(m->get_num_stored_elements(),
                  (4 * (2 * 1 + 2 + 1)) + (5 * (2 * 3 + 1 + 1)));
        ASSERT_EQ(m->get_num_stored_elements(0), (4 * (2 * 1 + 2 + 1)));
        ASSERT_EQ(m->get_num_stored_elements(1), (5 * (2 * 3 + 1 + 1)));

        EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 0, 0), value_type{2.0});
        EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 1, 0), value_type{4.0});
        EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 0, 1), value_type{3.0});
        EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 1, 1), value_type{1.0});
        EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 2, 1), value_type{0.0});
        EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 0, 2),
                  value_type{11.0});
        EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 1, 2), value_type{7.0});
        EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 2, 2), value_type{9.0});
        EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 3, 2), value_type{8.0});
        EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 1, 3),
                  value_type{13.0});
        EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 2, 3), value_type{8.0});
        EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 3, 3), value_type{4.0});

        EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 0, 0), value_type{9.0});
        EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 1, 0), value_type{0.0});
        EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 2, 0),
                  value_type{12.0});
        EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 3, 0), value_type{2.0});
        EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 0, 1), value_type{8.0});
        EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 1, 1), value_type{3.0});
        EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 2, 1), value_type{7.0});
        EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 3, 1),
                  value_type{15.0});
        EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 4, 1), value_type{4.0});
        EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 1, 2), value_type{5.0});
        EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 2, 2), value_type{1.0});
        EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 3, 2), value_type{8.0});
        EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 4, 2),
                  value_type{19.0});
        EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 2, 3), value_type{4.0});
        EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 3, 3), value_type{2.0});
        EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 4, 3), value_type{6.0});
        EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 3, 4), value_type{1.0});
        EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 4, 4), value_type{3.0});
    }

    static void assert_empty(gko::matrix::BatchBand<value_type>* m)
    {
        ASSERT_EQ(m->get_num_batch_entries(), 0);
        ASSERT_EQ(m->get_num_stored_elements(), 0);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<gko::matrix::BatchBand<value_type>> mtx;
};

TYPED_TEST_SUITE(BatchBand, gko::test::ValueTypes);


TYPED_TEST(BatchBand, CanBeEmpty)
{
    auto empty = gko::matrix::BatchBand<TypeParam>::create(this->exec);
    this->assert_empty(empty.get());
}


TYPED_TEST(BatchBand, ReturnsNullValuesArrayWhenEmpty)
{
    auto empty = gko::matrix::BatchBand<TypeParam>::create(this->exec);

    ASSERT_EQ(empty->get_const_band_array(), nullptr);
}

TYPED_TEST(BatchBand, CanBeConstructedWith_Size_KL_LU)
{
    using size_type = gko::size_type;
    auto m = gko::matrix::BatchBand<TypeParam>::create(
        this->exec,
        std::vector<gko::dim<2>>{gko::dim<2>{3, 3}, gko::dim<2>{4, 4}},
        std::vector<size_type>{1, 2}, std::vector<size_type>{2, 3});

    ASSERT_EQ(m->get_num_batch_entries(), 2);
    ASSERT_EQ(m->get_size().at(0), gko::dim<2>(3, 3));
    ASSERT_EQ(m->get_size().at(1), gko::dim<2>(4, 4));

    ASSERT_EQ(m->get_num_subdiagonals().at(0), 1);
    ASSERT_EQ(m->get_num_subdiagonals().at(1), 2);
    ASSERT_EQ(m->get_num_superdiagonals().at(0), 2);
    ASSERT_EQ(m->get_num_superdiagonals().at(1), 3);
    ASSERT_EQ(m->get_num_stored_elements(),
              (3 * (2 * 1 + 2 + 1)) + (4 * (2 * 2 + 3 + 1)));
    ASSERT_EQ(m->get_num_stored_elements(0), (3 * (2 * 1 + 2 + 1)));
    ASSERT_EQ(m->get_num_stored_elements(1), (4 * (2 * 2 + 3 + 1)));
}

TYPED_TEST(BatchBand, KnowsItsSizeAndValues)
{
    this->assert_equal_to_original_mtx(this->mtx.get());
}


TYPED_TEST(BatchBand, CanBeCopied)
{
    auto mtx_copy = gko::matrix::BatchBand<TypeParam>::create(this->exec);
    mtx_copy->copy_from(this->mtx.get());
    this->assert_equal_to_original_mtx(this->mtx.get());
    this->assert_equal_to_original_mtx(mtx_copy.get());
}


TYPED_TEST(BatchBand, CanBeMoved)
{
    auto mtx_copy = gko::matrix::BatchBand<TypeParam>::create(this->exec);
    mtx_copy->copy_from(std::move(this->mtx));
    this->assert_equal_to_original_mtx(mtx_copy.get());
}


TYPED_TEST(BatchBand, CanBeCloned)
{
    auto mtx_clone = this->mtx->clone();
    this->assert_equal_to_original_mtx(
        dynamic_cast<decltype(this->mtx.get())>(mtx_clone.get()));
}


TYPED_TEST(BatchBand, CanBeCleared)
{
    this->mtx->clear();
    this->assert_empty(this->mtx.get());
}

TYPED_TEST(BatchBand, CanBeConstructedFromExistingData)
{
    using value_type = typename TestFixture::value_type;
    using size_type = gko::size_type;

    // clang-format off
    /*
            General banded matrix A, where N = 6, KL = 2, KU = 1

            A_0 =    1   3    0   0   0   0 
                     2   9    3   0   0   0
                     1   8    4  -8   0   0
                     0   2    7   3   9   0
                     0   0    8   1   2   3
                     0   0    0   9   5   6


            Stored as band array AB:

            AB_0 =   *   *    *    +   +   +
                     *   *    +    +   +   +
                     *   3    3   -8   9   3
                     1   9    4    3   2   6
                     2   8    7    1   5   *
                     1   2    8    9   *   *

            General banded matrix A, where N = 4, KL = 2, KU = 2

            A_1 =    2  3  5   0
                     1  9  8   7
                     5  7  -1  2
                     0  2  0   9 

            AB_1 =  
                    *  *   *  *
                    *  *   *  *
                    *  *   5  7
                    *  3   8  2
                    2  9  -1  9
                    1  7   0  *
                    5  2   *  *   

    */
    
    value_type band_dense_col_major_arr[] = { 
        gko::nan<value_type>(), gko::nan<value_type>(), gko::nan<value_type>(), 1.0, 2.0, 1.0, //first col of 1st
        gko::nan<value_type>(), gko::nan<value_type>(), 3.0, 9.0, 8.0, 2.0,  //second col of 1st
        gko::nan<value_type>(), gko::nan<value_type>(), 3.0, 4.0, 7.0, 8.0, //third col of 1st
        gko::nan<value_type>(), gko::nan<value_type>(), -8.0, 3.0, 1.0, 9.0,  //fourth col of 1st
        gko::nan<value_type>(), gko::nan<value_type>(), 9.0, 2.0, 5.0,  gko::nan<value_type>(), //fifth col of 1st
        gko::nan<value_type>(), gko::nan<value_type>(), 3.0, 6.0, gko::nan<value_type>(), gko::nan<value_type>(), //sixth col of 1st
        gko::nan<value_type>(), gko::nan<value_type>(), gko::nan<value_type>(),gko::nan<value_type>(), 2.0, 1.0, 5.0 , //first col of 2nd
        gko::nan<value_type>(), gko::nan<value_type>(), gko::nan<value_type>(), 3.0, 9.0, 7.0, 2.0, //second col of 2nd
        gko::nan<value_type>(),  gko::nan<value_type>(), 5.0,  8.0, -1.0, 0.0,  gko::nan<value_type>(), //third col of 2nd 
        gko::nan<value_type>(),  gko::nan<value_type>(), 7.0,  2.0, 9.0,  gko::nan<value_type>(),  gko::nan<value_type>() //fourth col of 2nd
    };

    // clang-format on

    auto m = gko::matrix::BatchBand<TypeParam>::create(
        this->exec,
        std::vector<gko::dim<2>>{gko::dim<2>{6, 6}, gko::dim<2>{4, 4}},
        std::vector<size_type>{2, 2}, std::vector<size_type>{1, 2},
        gko::array<value_type>::view(this->exec, 64, band_dense_col_major_arr));

    ASSERT_EQ(m->get_const_band_array(), band_dense_col_major_arr);

    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 0, 0), value_type{1.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 1, 0), value_type{2.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 2, 0), value_type{1.0});

    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 0, 1), value_type{3.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 1, 1), value_type{9.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 2, 1), value_type{8.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 3, 1), value_type{2.0});

    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 1, 2), value_type{3.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 2, 2), value_type{4.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 3, 2), value_type{7.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 4, 2), value_type{8.0});

    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 2, 3), value_type{-8.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 3, 3), value_type{3.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 4, 3), value_type{1.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 5, 3), value_type{9.0});

    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 3, 4), value_type{9.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 4, 4), value_type{2.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 5, 4), value_type{5.0});

    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 4, 5), value_type{3.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 5, 5), value_type{6.0});


    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 0, 0), value_type{2.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 1, 0), value_type{1.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 2, 0), value_type{5.0});

    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 0, 1), value_type{3.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 1, 1), value_type{9.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 2, 1), value_type{7.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 3, 1), value_type{2.0});

    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 0, 2), value_type{5.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 1, 2), value_type{8.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 2, 2), value_type{-1.0});

    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 1, 3), value_type{7.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 2, 3), value_type{2.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 3, 3), value_type{9.0});
}


TYPED_TEST(BatchBand, CanBeConstructedFromExistingConstData)
{
    using value_type = typename TestFixture::value_type;
    using size_type = gko::size_type;

    // clang-format off
    /*
            General banded matrix A, where N = 6, KL = 2, KU = 1

            A_0 =    1   3    0   0   0   0 
                     2   9    3   0   0   0
                     1   8    4  -8   0   0
                     0   2    7   3   9   0
                     0   0    8   1   2   3
                     0   0    0   9   5   6


            band_arr_A_0 =  
                     *   *    *    +   +   +
                     *   *    +    +   +   +
                     *   3    3   -8   9   3
                     1   9    4    3   2   6
                     2   8    7    1   5   *
                     1   2    8    9   *   *

            General banded matrix A, where N = 4, KL = 2, KU = 2

            A_1 =    2  3  5   0
                     1  9  8   7
                     5  7  -1  2
                     0  2  0   9 

            band_arr_A_1 =  
                    *  *   *  *
                    *  *   *  *
                    *  *   5  7
                    *  3   8  2
                    2  9  -1  9
                    1  7   0  *
                    5  2   *  *   

    */
    
    value_type band_dense_col_major_arr[] = { 
        gko::nan<value_type>(), gko::nan<value_type>(), gko::nan<value_type>(), 1.0, 2.0, 1.0, //first col of 1st
        gko::nan<value_type>(), gko::nan<value_type>(), 3.0, 9.0, 8.0, 2.0,  //second col of 1st
        gko::nan<value_type>(), gko::nan<value_type>(), 3.0, 4.0, 7.0, 8.0, //third col of 1st
        gko::nan<value_type>(), gko::nan<value_type>(), -8.0, 3.0, 1.0, 9.0,  //fourth col of 1st
        gko::nan<value_type>(), gko::nan<value_type>(), 9.0, 2.0, 5.0,  gko::nan<value_type>(), //fifth col of 1st
        gko::nan<value_type>(), gko::nan<value_type>(), 3.0, 6.0, gko::nan<value_type>(), gko::nan<value_type>(), //sixth col of 1st
        gko::nan<value_type>(), gko::nan<value_type>(), gko::nan<value_type>(),gko::nan<value_type>(), 2.0, 1.0, 5.0 , //first col of 2nd
        gko::nan<value_type>(), gko::nan<value_type>(), gko::nan<value_type>(), 3.0, 9.0, 7.0, 2.0, //second col of 2nd
        gko::nan<value_type>(),  gko::nan<value_type>(), 5.0,  8.0, -1.0, 0.0,  gko::nan<value_type>(), //third col of 2nd 
        gko::nan<value_type>(),  gko::nan<value_type>(), 7.0,  2.0, 9.0,  gko::nan<value_type>(),  gko::nan<value_type>() //fourth col of 2nd
    };

    // clang-format on

    auto m = gko::matrix::BatchBand<TypeParam>::create_const(
        this->exec,
        std::vector<gko::dim<2>>{gko::dim<2>{6, 6}, gko::dim<2>{4, 4}},
        std::vector<size_type>{2, 2}, std::vector<size_type>{1, 2},
        gko::array<value_type>::const_view(this->exec, 64,
                                           band_dense_col_major_arr));

    ASSERT_EQ(m->get_const_band_array(), band_dense_col_major_arr);

    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 0, 0), value_type{1.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 1, 0), value_type{2.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 2, 0), value_type{1.0});

    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 0, 1), value_type{3.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 1, 1), value_type{9.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 2, 1), value_type{8.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 3, 1), value_type{2.0});

    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 1, 2), value_type{3.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 2, 2), value_type{4.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 3, 2), value_type{7.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 4, 2), value_type{8.0});

    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 2, 3), value_type{-8.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 3, 3), value_type{3.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 4, 3), value_type{1.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 5, 3), value_type{9.0});

    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 3, 4), value_type{9.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 4, 4), value_type{2.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 5, 4), value_type{5.0});

    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 4, 5), value_type{3.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 5, 5), value_type{6.0});


    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 0, 0), value_type{2.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 1, 0), value_type{1.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 2, 0), value_type{5.0});

    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 0, 1), value_type{3.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 1, 1), value_type{9.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 2, 1), value_type{7.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 3, 1), value_type{2.0});

    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 0, 2), value_type{5.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 1, 2), value_type{8.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 2, 2), value_type{-1.0});

    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 1, 3), value_type{7.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 2, 3), value_type{2.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 3, 3), value_type{9.0});
}

TYPED_TEST(BatchBand, CanBeConstructedFromBatchBandMatricesByDuplication)
{
    using value_type = typename TestFixture::value_type;
    using size_type = gko::size_type;

    // clang-format off
    /*
    
    A_0 : 
        3  4  6  0
        7  9  1  5
        0  8  2  4
        0  0  1  9

    band A_0 arr:
        *  *  *  *
        *  *  6  5
        *  4  1  4
        3  9  2  9
        7  8  1  * 

    A_1:  
        5  6  8  0
        1  2  4  6
        0  8  1  9
        0  0  8  9 

    Band A_1 arr:
        *  *  *  *
        *  *  8  6
        *  6  4  9
        5  2  1  9
        1  8  8  *  
    
    */
   
    value_type band_col_major_arr[] = {
    gko::nan<value_type>(), gko::nan<value_type>(), gko::nan<value_type>(), 3.0, 7.0, //1st col- 1st mat
    gko::nan<value_type>(),  gko::nan<value_type>(), 4.0, 9.0, 8.0, //2nd col- 1st mat
    gko::nan<value_type>(), 6.0, 1.0, 2.0, 1.0, //3rd col- 1st mat
    gko::nan<value_type>(), 5.0, 4.0, 9.0, gko::nan<value_type>(), //4th col- 1st mat
    gko::nan<value_type>(),  gko::nan<value_type>(),  gko::nan<value_type>(), 5.0, 1.0, //1st col - 2nd mat
    gko::nan<value_type>(),  gko::nan<value_type>(), 6.0, 2.0, 8.0, //2nd col- 2nd mat
    gko::nan<value_type>(), 8.0, 4.0, 1.0, 8.0, //3rd col- 2nd mat
    gko::nan<value_type>(), 6.0, 9.0, 9.0, gko::nan<value_type>() //4th col - 2nd mat
    };
    // clang-format on


    auto m = gko::matrix::BatchBand<value_type>::create(
        this->exec, gko::batch_dim<2>(2, gko::dim<2>{4, 4}),
        gko::batch_stride(2, 1), gko::batch_stride(2, 2),
        gko::array<value_type>::view(this->exec, 40, band_col_major_arr));

    auto bat_m_created_by_dupl =
        gko::matrix::BatchBand<value_type>::create(this->exec, 2, m.get());

    // clang-format off
    value_type band_col_major_arr_new[] = {
    gko::nan<value_type>(), gko::nan<value_type>(), gko::nan<value_type>(), 3.0, 7.0, //1st col- 1st mat
    gko::nan<value_type>(),  gko::nan<value_type>(), 4.0, 9.0, 8.0, //2nd col- 1st mat
    gko::nan<value_type>(), 6.0, 1.0, 2.0, 1.0, //3rd col- 1st mat
    gko::nan<value_type>(), 5.0, 4.0, 9.0, gko::nan<value_type>(), //4th col- 1st mat

    gko::nan<value_type>(),  gko::nan<value_type>(),  gko::nan<value_type>(), 5.0, 1.0, //1st col - 2nd mat
    gko::nan<value_type>(),  gko::nan<value_type>(), 6.0, 2.0, 8.0, //2nd col- 2nd mat
    gko::nan<value_type>(), 8.0, 4.0, 1.0, 8.0, //3rd col- 2nd mat
    gko::nan<value_type>(), 6.0, 9.0, 9.0, gko::nan<value_type>(), //4th col - 2nd mat

    gko::nan<value_type>(), gko::nan<value_type>(), gko::nan<value_type>(), 3.0, 7.0, //1st col- 3rd mat
    gko::nan<value_type>(),  gko::nan<value_type>(), 4.0, 9.0, 8.0, //2nd col- 3rd mat
    gko::nan<value_type>(), 6.0, 1.0, 2.0, 1.0, //3rd col- 3rd mat
    gko::nan<value_type>(), 5.0, 4.0, 9.0, gko::nan<value_type>(), //4th col- 3rd mat

    gko::nan<value_type>(),  gko::nan<value_type>(),  gko::nan<value_type>(), 5.0, 1.0, //1st col - 4th mat
    gko::nan<value_type>(),  gko::nan<value_type>(), 6.0, 2.0, 8.0, //2nd col- 4th mat
    gko::nan<value_type>(), 8.0, 4.0, 1.0, 8.0, //3rd col- 4th mat
    gko::nan<value_type>(), 6.0, 9.0, 9.0, gko::nan<value_type>(), //4th col - 4th mat

    };
    // clang-format on

    auto m_new = gko::matrix::BatchBand<value_type>::create(
        this->exec, gko::batch_dim<2>(4, gko::dim<2>{4, 4}),
        gko::batch_stride(4, 1), gko::batch_stride(4, 2),
        gko::array<value_type>::view(this->exec, 80, band_col_major_arr_new));

    GKO_ASSERT_BATCH_MTX_NEAR(bat_m_created_by_dupl.get(), m_new.get(), 1e-14);
}

TYPED_TEST(BatchBand, CanBeReadFromMatrixData)
{
    using value_type = typename TestFixture::value_type;
    auto m = gko::matrix::BatchBand<value_type>::create(this->exec);

    /*

        first matrix:
        2  4  0  0
        3  6  1  8
        0  0  9  7
        0  4  5  6

        second matrix:
        4  3  0
        3  7  0
        0  0  8

    */

    // clang-format off
    m->read({gko::matrix_data<value_type>{{4, 4},
                                         {{0, 0, 2.0},
                                          {0, 1, 4.0},
                                          {1, 0, 3.0},
                                          {1, 1, 6.0},
                                          {1, 2, 1.0},
                                          {1, 3, 8.0},
                                          {2, 2, 9.0},
                                          {2, 3, 7.0},
                                          {3, 1, 4.0}, 
                                          {3, 2, 5.0},
                                          {3, 3, 6.0}}},
             gko::matrix_data<TypeParam>{{3, 3},
                                         {{0, 0, 4.0},
                                          {0, 1, 3.0},
                                          {1, 0, 3.0},
                                          {1, 1, 7.0},
                                          {2, 2, 8.0}}}});
    // clang-format on

    ASSERT_EQ(m->get_size().at(0), gko::dim<2>(4, 4));
    ASSERT_EQ(m->get_size().at(1), gko::dim<2>(3, 3));

    ASSERT_EQ(m->get_num_subdiagonals(),
              gko::batch_stride(std::vector<gko::size_type>{2, 1}));
    ASSERT_EQ(m->get_num_superdiagonals(),
              gko::batch_stride(std::vector<gko::size_type>{2, 1}));

    ASSERT_EQ(m->get_num_stored_elements(), 40);
    ASSERT_EQ(m->get_num_stored_elements(0), 28);
    ASSERT_EQ(m->get_num_stored_elements(1), 12);

    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 0, 0), value_type{2.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 1, 0), value_type{3.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 2, 0), value_type{0.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 0, 1), value_type{4.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 1, 1), value_type{6.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 2, 1), value_type{0.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 3, 1), value_type{4.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 0, 2), value_type{0.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 1, 2), value_type{1.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 2, 2), value_type{9.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 3, 2), value_type{5.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 1, 3), value_type{8.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 2, 3), value_type{7.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 3, 3), value_type{6.0});


    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 0, 0), value_type{4.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 1, 0), value_type{3.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 0, 1), value_type{3.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 1, 1), value_type{7.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 2, 1), value_type{0.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 1, 2), value_type{0.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 2, 2), value_type{8.0});
}


TYPED_TEST(BatchBand, CanBeReadFromMatrixAssemblyData)
{
    using value_type = typename TestFixture::value_type;
    auto m = gko::matrix::BatchBand<value_type>::create(this->exec);

    /*

        first matrix:
        2  4  0  0
        3  6  1  8
        0  0  9  7
        0  4  5  6

        second matrix:
        4  3  0
        3  7  0
        0  0  8

    */

    gko::matrix_assembly_data<value_type> data1(gko::dim<2>{4, 4});
    data1.set_value(0, 0, 2.0);
    data1.set_value(0, 1, 4.0);
    data1.set_value(1, 0, 3.0);
    data1.set_value(1, 1, 6.0);
    data1.set_value(1, 2, 1.0);
    data1.set_value(1, 3, 8.0);
    data1.set_value(2, 2, 9.0);
    data1.set_value(2, 3, 7.0);
    data1.set_value(3, 1, 4.0);
    data1.set_value(3, 2, 5.0);
    data1.set_value(3, 3, 6.0);

    gko::matrix_assembly_data<value_type> data2(gko::dim<2>{3, 3});
    data2.set_value(0, 0, 4.0);
    data2.set_value(0, 1, 3.0);
    data2.set_value(1, 0, 3.0);
    data2.set_value(1, 1, 7.0);
    data2.set_value(2, 2, 8.0);

    auto data = std::vector<gko::matrix_assembly_data<TypeParam>>{data1, data2};
    m->read(data);

    ASSERT_EQ(m->get_size().at(0), gko::dim<2>(4, 4));
    ASSERT_EQ(m->get_size().at(1), gko::dim<2>(3, 3));

    ASSERT_EQ(m->get_num_subdiagonals(),
              gko::batch_stride(std::vector<gko::size_type>{2, 1}));
    ASSERT_EQ(m->get_num_superdiagonals(),
              gko::batch_stride(std::vector<gko::size_type>{2, 1}));

    ASSERT_EQ(m->get_num_stored_elements(), 40);
    ASSERT_EQ(m->get_num_stored_elements(0), 28);
    ASSERT_EQ(m->get_num_stored_elements(1), 12);

    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 0, 0), value_type{2.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 1, 0), value_type{3.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 2, 0), value_type{0.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 0, 1), value_type{4.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 1, 1), value_type{6.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 2, 1), value_type{0.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 3, 1), value_type{4.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 0, 2), value_type{0.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 1, 2), value_type{1.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 2, 2), value_type{9.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 3, 2), value_type{5.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 1, 3), value_type{8.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 2, 3), value_type{7.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 3, 3), value_type{6.0});


    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 0, 0), value_type{4.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 1, 0), value_type{3.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 0, 1), value_type{3.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 1, 1), value_type{7.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 2, 1), value_type{0.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 1, 2), value_type{0.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 2, 2), value_type{8.0});
}

TYPED_TEST(BatchBand, CanBeReadFromMatrixDataWhenKLAndKUAreGivenByTheUser)
{
    using value_type = typename TestFixture::value_type;
    auto m = gko::matrix::BatchBand<value_type>::create(this->exec);

    /*

        first matrix:
        2  4  0  0
        3  6  1  8
        0  0  9  7
        0  4  5  6

        second matrix:
        4  3  0
        3  7  0
        0  0  8

    */

    // clang-format off
    m->read({gko::matrix_data<value_type>{{4, 4},
                                         {{0, 0, 2.0},
                                          {0, 1, 4.0},
                                          {1, 0, 3.0},
                                          {1, 1, 6.0},
                                          {1, 2, 1.0},
                                          {1, 3, 8.0},
                                          {2, 2, 9.0},
                                          {2, 3, 7.0},
                                          {3, 1, 4.0}, 
                                          {3, 2, 5.0},
                                          {3, 3, 6.0}}},
             gko::matrix_data<TypeParam>{{3, 3},
                                         {{0, 0, 4.0},
                                          {0, 1, 3.0},
                                          {1, 0, 3.0},
                                          {1, 1, 7.0},
                                          {2, 2, 8.0}}}}, 
                                          gko::batch_stride(std::vector<gko::size_type>{2, 1}), 
                                          gko::batch_stride(std::vector<gko::size_type>{2, 1}));
    // clang-format on

    ASSERT_EQ(m->get_size().at(0), gko::dim<2>(4, 4));
    ASSERT_EQ(m->get_size().at(1), gko::dim<2>(3, 3));

    ASSERT_EQ(m->get_num_subdiagonals(),
              gko::batch_stride(std::vector<gko::size_type>{2, 1}));
    ASSERT_EQ(m->get_num_superdiagonals(),
              gko::batch_stride(std::vector<gko::size_type>{2, 1}));

    ASSERT_EQ(m->get_num_stored_elements(), 40);
    ASSERT_EQ(m->get_num_stored_elements(0), 28);
    ASSERT_EQ(m->get_num_stored_elements(1), 12);

    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 0, 0), value_type{2.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 1, 0), value_type{3.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 2, 0), value_type{0.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 0, 1), value_type{4.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 1, 1), value_type{6.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 2, 1), value_type{0.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 3, 1), value_type{4.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 0, 2), value_type{0.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 1, 2), value_type{1.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 2, 2), value_type{9.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 3, 2), value_type{5.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 1, 3), value_type{8.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 2, 3), value_type{7.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(0, 3, 3), value_type{6.0});


    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 0, 0), value_type{4.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 1, 0), value_type{3.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 0, 1), value_type{3.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 1, 1), value_type{7.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 2, 1), value_type{0.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 1, 2), value_type{0.0});
    EXPECT_EQ(m->at_in_reference_to_dense_layout(1, 2, 2), value_type{8.0});
}


TYPED_TEST(BatchBand, GeneratesCorrectMatrixData)
{
    using value_type = typename TestFixture::value_type;
    using tpl = typename gko::matrix_data<TypeParam>::nonzero_type;
    std::vector<gko::matrix_data<value_type>> data;

    this->mtx->write(data);

    ASSERT_EQ(data[0].size, gko::dim<2>(4, 4));

    ASSERT_EQ(data[0].nonzeros.size(), 11);
    EXPECT_EQ(data[0].nonzeros[0], tpl(0, 0, value_type{2.0}));
    EXPECT_EQ(data[0].nonzeros[1], tpl(0, 1, value_type{3.0}));
    EXPECT_EQ(data[0].nonzeros[2], tpl(0, 2, value_type{11.0}));
    EXPECT_EQ(data[0].nonzeros[3], tpl(1, 0, value_type{4.0}));
    EXPECT_EQ(data[0].nonzeros[4], tpl(1, 1, value_type{1.0}));
    EXPECT_EQ(data[0].nonzeros[5], tpl(1, 2, value_type{7.0}));
    EXPECT_EQ(data[0].nonzeros[6], tpl(1, 3, value_type{13.0}));
    EXPECT_EQ(data[0].nonzeros[7], tpl(2, 2, value_type{9.0}));
    EXPECT_EQ(data[0].nonzeros[8], tpl(2, 3, value_type{8.0}));
    EXPECT_EQ(data[0].nonzeros[9], tpl(3, 2, value_type{8.0}));
    EXPECT_EQ(data[0].nonzeros[10], tpl(3, 3, value_type{4.0}));

    ASSERT_EQ(data[1].size, gko::dim<2>(5, 5));
    ASSERT_EQ(data[1].nonzeros.size(), 17);

    EXPECT_EQ(data[1].nonzeros[0], tpl(0, 0, value_type{9.0}));
    EXPECT_EQ(data[1].nonzeros[1], tpl(0, 1, value_type{8.0}));

    EXPECT_EQ(data[1].nonzeros[2], tpl(1, 1, value_type{3.0}));
    EXPECT_EQ(data[1].nonzeros[3], tpl(1, 2, value_type{5.0}));

    EXPECT_EQ(data[1].nonzeros[4], tpl(2, 0, value_type{12.0}));
    EXPECT_EQ(data[1].nonzeros[5], tpl(2, 1, value_type{7.0}));
    EXPECT_EQ(data[1].nonzeros[6], tpl(2, 2, value_type{1.0}));
    EXPECT_EQ(data[1].nonzeros[7], tpl(2, 3, value_type{4.0}));

    EXPECT_EQ(data[1].nonzeros[8], tpl(3, 0, value_type{2.0}));
    EXPECT_EQ(data[1].nonzeros[9], tpl(3, 1, value_type{15.0}));
    EXPECT_EQ(data[1].nonzeros[10], tpl(3, 2, value_type{8.0}));
    EXPECT_EQ(data[1].nonzeros[11], tpl(3, 3, value_type{2.0}));
    EXPECT_EQ(data[1].nonzeros[12], tpl(3, 4, value_type{1.0}));

    EXPECT_EQ(data[1].nonzeros[13], tpl(4, 1, value_type{4.0}));
    EXPECT_EQ(data[1].nonzeros[14], tpl(4, 2, value_type{19.0}));
    EXPECT_EQ(data[1].nonzeros[15], tpl(4, 3, value_type{6.0}));
    EXPECT_EQ(data[1].nonzeros[16], tpl(4, 4, value_type{3.0}));
}

TYPED_TEST(BatchBand, ThrowsOnRectangularMatrix)
{
    ASSERT_THROW(gko::matrix::BatchBand<TypeParam>::create(
                     this->exec, gko::batch_dim<2>(2, gko::dim<2>{3, 5}),
                     gko::batch_stride(2, 2), gko::batch_stride(2, 3)),
                 gko::DimensionMismatch);
}

}  // namespace
