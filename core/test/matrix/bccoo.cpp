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

#include <ginkgo/core/matrix/bccoo.hpp>


#include <gtest/gtest.h>


#include "core/base/unaligned_access.hpp"
#include "core/matrix/bccoo_aux_structs.hpp"
#include "core/test/utils.hpp"


using namespace gko::matrix::bccoo;


namespace {


constexpr static int BCCOO_BLOCK_SIZE_TESTED = 1;
constexpr static int BCCOO_BLOCK_SIZE_COPIED = 3;


template <typename ValueIndexType>
class Bccoo : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Bccoo<value_type, index_type>;

    Bccoo()
        : exec(gko::ReferenceExecutor::create()),
          mtx_ind(gko::matrix::Bccoo<value_type, index_type>::create(
              exec, index_type{BCCOO_BLOCK_SIZE_TESTED},
              compression::individual)),
          mtx_grp(gko::matrix::Bccoo<value_type, index_type>::create(
              exec, index_type{BCCOO_BLOCK_SIZE_TESTED}, compression::group))
    {
        mtx_ind->read({{2, 3},
                       {
                           {0, 0, 1.0},
                           {0, 1, 3.0},
                           {0, 2, 2.0},
                           {1, 1, 5.0},
                       }});
        mtx_grp->read({{2, 3},
                       {
                           {0, 0, 1.0},
                           {0, 1, 3.0},
                           {0, 2, 2.0},
                           {1, 1, 5.0},
                       }});
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx_ind;
    std::unique_ptr<Mtx> mtx_grp;

    void assert_equal_to_original_mtx_ind(const Mtx* m)
    {
        auto start_rows = m->get_const_start_rows();
        auto block_offsets = m->get_const_block_offsets();
        auto compressed_data = m->get_const_compressed_data();

        ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_num_stored_elements(), 4);

        index_type block_size = m->get_block_size();

        index_type row = {};
        gko::size_type offset = {};
        for (index_type i = 0; i < m->get_num_blocks(); i++) {
            EXPECT_EQ(start_rows[i], row);
            EXPECT_EQ(block_offsets[i], offset);
            auto elms = std::min(block_size, 4 - i * block_size);
            row += ((block_size == 1) && (i == 2)) || (block_size == 3);
            offset += (1 + sizeof(value_type)) * elms +
                      (((block_size == 2) || (block_size >= 4)) &&
                       (i + block_size > 2));
        }
        EXPECT_EQ(block_offsets[m->get_num_blocks()], offset);

        index_type ind = {};

        EXPECT_EQ(compressed_data[ind], 0x00);
        ind++;
        EXPECT_EQ(get_value_compressed_data<value_type>(compressed_data, ind),
                  value_type{1.0});
        ind += sizeof(value_type);

        EXPECT_EQ(compressed_data[ind], 0x01);
        ind++;
        EXPECT_EQ(get_value_compressed_data<value_type>(compressed_data, ind),
                  value_type{3.0});
        ind += sizeof(value_type);

        if (block_size < 3) {
            EXPECT_EQ(compressed_data[ind], 0x02);
        } else {
            EXPECT_EQ(compressed_data[ind], 0x01);
        }
        ind++;
        EXPECT_EQ(get_value_compressed_data<value_type>(compressed_data, ind),
                  value_type{2.0});
        ind += sizeof(value_type);

        if ((block_size == 2) || (block_size >= 4)) {
            EXPECT_EQ(compressed_data[ind], 0xFF);
            ind++;
        }

        EXPECT_EQ(compressed_data[ind], 0x01);
        ind++;
        EXPECT_EQ(get_value_compressed_data<value_type>(compressed_data, ind),
                  value_type{5.0});
        ind += sizeof(value_type);
    }

    void assert_equal_to_original_mtx_grp(const Mtx* m)
    {
        auto start_rows = m->get_const_start_rows();
        auto start_cols = m->get_const_start_cols();
        auto compression_types = m->get_const_compression_types();
        auto block_offsets = m->get_const_block_offsets();
        auto compressed_data = m->get_const_compressed_data();

        ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_num_stored_elements(), 4);

        index_type block_size = m->get_block_size();

        index_type row = {};
        index_type col = {};
        gko::uint8 type = ((block_size >= 4) ? 3 : 2);
        gko::size_type offset = {};
        for (index_type i = 0; i < m->get_num_blocks(); i++) {
            auto elms = std::min(block_size, 4 - i * block_size);
            row = ((i > 0) && (i + block_size) == 4) ? 1 : 0;
            col = i % 3 + i / 3;
            type = (((block_size == 2) || (block_size >= 4)) &&
                    (i + block_size > 2))
                       ? (type_mask_cols_8bits | type_mask_rows_multiple)
                       : type_mask_cols_8bits;
            EXPECT_EQ(start_rows[i], row);
            EXPECT_EQ(start_cols[i], col);
            EXPECT_EQ(compression_types[i], type);
            EXPECT_EQ(block_offsets[i], offset);
            offset += (1 + sizeof(value_type)) * elms +
                      (((block_size == 2) || (block_size >= 4)) &&
                       (i + block_size > 2)) *
                          elms;
        }

        index_type ind = {};

        ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_num_stored_elements(), 4);

        switch (block_size) {
        case 1:
            // block 0
            EXPECT_EQ(compressed_data[ind], 0x00);
            ind++;
            EXPECT_EQ(
                get_value_compressed_data<value_type>(compressed_data, ind),
                value_type{1.0});
            ind += sizeof(value_type);

            // block 1
            EXPECT_EQ(compressed_data[ind], 0x00);
            ind++;
            EXPECT_EQ(
                get_value_compressed_data<value_type>(compressed_data, ind),
                value_type{3.0});
            ind += sizeof(value_type);

            // block 2
            EXPECT_EQ(compressed_data[ind], 0x00);
            ind++;
            EXPECT_EQ(
                get_value_compressed_data<value_type>(compressed_data, ind),
                value_type{2.0});
            ind += sizeof(value_type);

            // block 3
            EXPECT_EQ(compressed_data[ind], 0x00);
            ind++;
            EXPECT_EQ(
                get_value_compressed_data<value_type>(compressed_data, ind),
                value_type{5.0});
            ind += sizeof(value_type);

            break;
        case 2:
            // block 0
            EXPECT_EQ(compressed_data[ind], 0x00);
            ind++;
            EXPECT_EQ(compressed_data[ind], 0x01);
            ind++;

            EXPECT_EQ(
                get_value_compressed_data<value_type>(compressed_data, ind),
                value_type{1.0});
            ind += sizeof(value_type);
            EXPECT_EQ(
                get_value_compressed_data<value_type>(compressed_data, ind),
                value_type{3.0});
            ind += sizeof(value_type);

            // block 1
            EXPECT_EQ(compressed_data[ind], 0x00);
            ind++;
            EXPECT_EQ(compressed_data[ind], 0x01);
            ind++;

            EXPECT_EQ(compressed_data[ind], 0x01);
            ind++;
            EXPECT_EQ(compressed_data[ind], 0x00);
            ind++;

            EXPECT_EQ(
                get_value_compressed_data<value_type>(compressed_data, ind),
                value_type{2.0});
            ind += sizeof(value_type);
            EXPECT_EQ(
                get_value_compressed_data<value_type>(compressed_data, ind),
                value_type{5.0});
            ind += sizeof(value_type);

            break;
        case 3:
            // block 0
            EXPECT_EQ(compressed_data[ind], 0x00);
            ind++;
            EXPECT_EQ(compressed_data[ind], 0x01);
            ind++;
            EXPECT_EQ(compressed_data[ind], 0x02);
            ind++;

            EXPECT_EQ(
                get_value_compressed_data<value_type>(compressed_data, ind),
                value_type{1.0});
            ind += sizeof(value_type);
            EXPECT_EQ(
                get_value_compressed_data<value_type>(compressed_data, ind),
                value_type{3.0});
            ind += sizeof(value_type);
            EXPECT_EQ(
                get_value_compressed_data<value_type>(compressed_data, ind),
                value_type{2.0});
            ind += sizeof(value_type);

            // block 1
            EXPECT_EQ(compressed_data[ind], 0x00);
            ind++;
            EXPECT_EQ(
                get_value_compressed_data<value_type>(compressed_data, ind),
                value_type{5.0});
            ind += sizeof(value_type);

            break;
        default:
            // block 0
            EXPECT_EQ(compressed_data[ind], 0x00);
            ind++;
            EXPECT_EQ(compressed_data[ind], 0x00);
            ind++;
            EXPECT_EQ(compressed_data[ind], 0x00);
            ind++;
            EXPECT_EQ(compressed_data[ind], 0x01);
            ind++;

            EXPECT_EQ(compressed_data[ind], 0x00);
            ind++;
            EXPECT_EQ(compressed_data[ind], 0x01);
            ind++;
            EXPECT_EQ(compressed_data[ind], 0x02);
            ind++;
            EXPECT_EQ(compressed_data[ind], 0x01);
            ind++;

            EXPECT_EQ(
                get_value_compressed_data<value_type>(compressed_data, ind),
                value_type{1.0});
            ind += sizeof(value_type);
            EXPECT_EQ(
                get_value_compressed_data<value_type>(compressed_data, ind),
                value_type{3.0});
            ind += sizeof(value_type);
            EXPECT_EQ(
                get_value_compressed_data<value_type>(compressed_data, ind),
                value_type{2.0});
            ind += sizeof(value_type);
            EXPECT_EQ(
                get_value_compressed_data<value_type>(compressed_data, ind),
                value_type{5.0});
            ind += sizeof(value_type);

            break;
        }
    }

    void assert_empty(const Mtx* m)
    {
        ASSERT_EQ(m->get_size(), gko::dim<2>(0, 0));
        ASSERT_EQ(m->get_compression(), compression::def_value);
        ASSERT_EQ(m->get_num_stored_elements(), 0);
        ASSERT_EQ(m->get_block_size(), 0);
        ASSERT_EQ(m->get_num_blocks(), 0);
        ASSERT_EQ(m->get_num_bytes(), 0);
        ASSERT_EQ(m->get_const_start_rows(), nullptr);
        ASSERT_EQ(m->get_const_start_cols(), nullptr);
        ASSERT_EQ(m->get_const_compression_types(), nullptr);
        ASSERT_EQ(m->get_const_block_offsets(), nullptr);
        ASSERT_EQ(m->get_const_compressed_data(), nullptr);
    }

    void assert_empty_ind(const Mtx* m)
    {
        ASSERT_EQ(m->get_size(), gko::dim<2>(0, 0));
        ASSERT_EQ(m->get_compression(), compression::individual);
        ASSERT_EQ(m->get_num_stored_elements(), 0);
        ASSERT_EQ(m->get_num_blocks(), 0);
        ASSERT_EQ(m->get_num_bytes(), 0);
        ASSERT_EQ(m->get_const_start_rows(), nullptr);
        ASSERT_EQ(m->get_const_start_cols(), nullptr);
        ASSERT_EQ(m->get_const_compression_types(), nullptr);
        ASSERT_EQ(m->get_const_block_offsets(), nullptr);
        ASSERT_EQ(m->get_const_compressed_data(), nullptr);
    }

    void assert_empty_grp(const Mtx* m)
    {
        ASSERT_EQ(m->get_size(), gko::dim<2>(0, 0));
        ASSERT_EQ(m->get_compression(), compression::group);
        ASSERT_EQ(m->get_num_stored_elements(), 0);
        ASSERT_EQ(m->get_num_blocks(), 0);
        ASSERT_EQ(m->get_num_bytes(), 0);
        ASSERT_EQ(m->get_const_start_rows(), nullptr);
        ASSERT_EQ(m->get_const_start_cols(), nullptr);
        ASSERT_EQ(m->get_const_compression_types(), nullptr);
        ASSERT_EQ(m->get_const_block_offsets(), nullptr);
        ASSERT_EQ(m->get_const_compressed_data(), nullptr);
    }
};

TYPED_TEST_SUITE(Bccoo, gko::test::ValueIndexTypes);


TYPED_TEST(Bccoo, KnowsItsSizeInd)
{
    ASSERT_EQ(this->mtx_ind->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(this->mtx_ind->get_num_stored_elements(), 4);
}


TYPED_TEST(Bccoo, KnowsItsSizeGrp)
{
    ASSERT_EQ(this->mtx_grp->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(this->mtx_grp->get_num_stored_elements(), 4);
}


TYPED_TEST(Bccoo, ContainsCorrectDataInd)
{
    this->assert_equal_to_original_mtx_ind(this->mtx_ind.get());
}


TYPED_TEST(Bccoo, ContainsCorrectDataGrp)
{
    this->assert_equal_to_original_mtx_grp(this->mtx_grp.get());
}


TYPED_TEST(Bccoo, CanBeEmpty)
{
    using Mtx = typename TestFixture::Mtx;
    auto mtx_ind = Mtx::create(this->exec);

    this->assert_empty(mtx_ind.get());
}


TYPED_TEST(Bccoo, CanBeEmptyInd)
{
    using Mtx = typename TestFixture::Mtx;
    auto mtx_ind = Mtx::create(this->exec, BCCOO_BLOCK_SIZE_TESTED,
                               compression::individual);

    this->assert_empty_ind(mtx_ind.get());
}


TYPED_TEST(Bccoo, CanBeEmptyGrp)
{
    using Mtx = typename TestFixture::Mtx;
    auto mtx_grp =
        Mtx::create(this->exec, BCCOO_BLOCK_SIZE_TESTED, compression::group);

    this->assert_empty_grp(mtx_grp.get());
}

TYPED_TEST(Bccoo, CanBeCreatedFromExistingDataInd)
{
    // Name the involved datatypes
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    // Declare the variables
    const index_type block_size = 10;
    const gko::size_type num_bytes = 6 + 4 * sizeof(value_type);
    index_type ind = {};
    gko::uint8 compressed_data[num_bytes] = {};
    gko::size_type block_offsets[] = {0, num_bytes};
    index_type start_rows[] = {0};
    // Fill the vectors
    compressed_data[ind++] = 0x00;
    set_value_compressed_data<value_type>(compressed_data, ind, 1.0);
    ind += sizeof(value_type);
    compressed_data[ind++] = 0x01;
    set_value_compressed_data<value_type>(compressed_data, ind, 2.0);
    ind += sizeof(value_type);
    compressed_data[ind++] = 0xFF;
    compressed_data[ind++] = 0x01;
    set_value_compressed_data<value_type>(compressed_data, ind, 3.0);
    ind += sizeof(value_type);
    compressed_data[ind++] = 0xFF;
    compressed_data[ind++] = 0x00;
    set_value_compressed_data<value_type>(compressed_data, ind, 4.0);
    ind += sizeof(value_type);

    auto mtx_ind = gko::matrix::Bccoo<value_type, index_type>::create(
        this->exec, gko::dim<2>{3, 2},
        gko::array<gko::uint8>::view(this->exec, num_bytes, compressed_data),
        gko::array<gko::size_type>::view(this->exec, 2, block_offsets),
        gko::array<index_type>::view(this->exec, 1, start_rows), 4, block_size);

    ASSERT_EQ(mtx_ind->get_num_stored_elements(), 4);
    ASSERT_EQ(mtx_ind->get_block_size(), block_size);
    ASSERT_EQ(mtx_ind->get_const_block_offsets(), block_offsets);
    ASSERT_EQ(mtx_ind->get_const_start_rows(), start_rows);
}


TYPED_TEST(Bccoo, CanBeCreatedFromExistingDataGrp)
{
    // Name the involved datatypes
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    // Declare the variables
    const index_type block_size = 10;
    const gko::size_type num_bytes = 4 + 4 + 4 * sizeof(value_type);
    index_type ind = {};
    gko::uint8 compressed_data[num_bytes] = {};
    gko::size_type block_offsets[] = {0, num_bytes};
    gko::uint8 compression_types[] = {3};
    index_type start_cols[] = {0};
    index_type start_rows[] = {0};
    // Fill the rows
    compressed_data[ind++] = 0x00;
    compressed_data[ind++] = 0x00;
    compressed_data[ind++] = 0x01;
    compressed_data[ind++] = 0x02;
    // Fill the colums
    compressed_data[ind++] = 0x00;
    compressed_data[ind++] = 0x01;
    compressed_data[ind++] = 0x01;
    compressed_data[ind++] = 0x00;
    // Fill the values
    set_value_compressed_data<value_type>(compressed_data, ind, 1.0);
    ind += sizeof(value_type);
    set_value_compressed_data<value_type>(compressed_data, ind, 2.0);
    ind += sizeof(value_type);
    set_value_compressed_data<value_type>(compressed_data, ind, 3.0);
    ind += sizeof(value_type);
    set_value_compressed_data<value_type>(compressed_data, ind, 4.0);
    ind += sizeof(value_type);

    auto mtx_grp = gko::matrix::Bccoo<value_type, index_type>::create(
        this->exec, gko::dim<2>{3, 2},
        gko::array<gko::uint8>::view(this->exec, num_bytes, compressed_data),
        gko::array<gko::size_type>::view(this->exec, 2, block_offsets),
        gko::array<gko::uint8>::view(this->exec, 1, compression_types),
        gko::array<index_type>::view(this->exec, 1, start_cols),
        gko::array<index_type>::view(this->exec, 1, start_rows), 4, block_size);

    ASSERT_EQ(mtx_grp->get_num_stored_elements(), 4);
    ASSERT_EQ(mtx_grp->get_block_size(), block_size);
    ASSERT_EQ(mtx_grp->get_const_block_offsets(), block_offsets);
    ASSERT_EQ(mtx_grp->get_const_compression_types(), compression_types);
    ASSERT_EQ(mtx_grp->get_const_start_cols(), start_cols);
    ASSERT_EQ(mtx_grp->get_const_start_rows(), start_rows);
}


TYPED_TEST(Bccoo, CanBeCopiedIndInd)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto copy = Mtx::create(this->exec, index_type{BCCOO_BLOCK_SIZE_COPIED},
                            compression::individual);

    copy->copy_from(this->mtx_ind.get());

    this->assert_equal_to_original_mtx_ind(this->mtx_ind.get());
    set_value_compressed_data<value_type>(this->mtx_ind->get_compressed_data(),
                                          2 + sizeof(value_type), 5.0);
    this->assert_equal_to_original_mtx_ind(copy.get());
}


TYPED_TEST(Bccoo, CanBeCopiedIndGrp)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto copy = Mtx::create(this->exec, index_type{BCCOO_BLOCK_SIZE_COPIED},
                            compression::group);

    copy->copy_from(this->mtx_ind.get());

    this->assert_equal_to_original_mtx_ind(this->mtx_ind.get());
    set_value_compressed_data<value_type>(this->mtx_ind->get_compressed_data(),
                                          2 + sizeof(value_type), 5.0);
    this->assert_equal_to_original_mtx_grp(copy.get());
}


TYPED_TEST(Bccoo, CanBeCopiedGrpInd)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto copy = Mtx::create(this->exec, index_type{BCCOO_BLOCK_SIZE_COPIED},
                            compression::individual);

    copy->copy_from(this->mtx_grp.get());

    this->assert_equal_to_original_mtx_grp(this->mtx_grp.get());
    set_value_compressed_data<value_type>(
        this->mtx_grp->get_compressed_data(),
        this->mtx_grp->get_num_bytes() - sizeof(value_type), 5.0);
    this->assert_equal_to_original_mtx_ind(copy.get());
}


TYPED_TEST(Bccoo, CanBeCopiedGrpGrp)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto copy = Mtx::create(this->exec, index_type{BCCOO_BLOCK_SIZE_COPIED},
                            compression::group);

    copy->copy_from(this->mtx_grp.get());

    this->assert_equal_to_original_mtx_grp(this->mtx_grp.get());
    set_value_compressed_data<value_type>(
        this->mtx_grp->get_compressed_data(),
        this->mtx_grp->get_num_bytes() - sizeof(value_type), 5.0);
    this->assert_equal_to_original_mtx_grp(copy.get());
}


TYPED_TEST(Bccoo, CanBeMovedInd)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto copy = Mtx::create(this->exec, index_type{BCCOO_BLOCK_SIZE_COPIED},
                            compression::individual);

    copy->move_from(this->mtx_ind);

    this->assert_equal_to_original_mtx_ind(copy.get());
}


TYPED_TEST(Bccoo, CanBeMovedGrp)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto copy = Mtx::create(this->exec, index_type{BCCOO_BLOCK_SIZE_COPIED},
                            compression::group);

    copy->move_from(this->mtx_grp);

    this->assert_equal_to_original_mtx_grp(copy.get());
}


TYPED_TEST(Bccoo, CanBeClonedInd)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;

    auto clone = this->mtx_ind->clone();

    this->assert_equal_to_original_mtx_ind(this->mtx_ind.get());

    set_value_compressed_data<value_type>(this->mtx_grp->get_compressed_data(),
                                          2 + sizeof(value_type), 5.0);
    this->assert_equal_to_original_mtx_ind(dynamic_cast<Mtx*>(clone.get()));
}


TYPED_TEST(Bccoo, CanBeClonedGrp)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto clone = this->mtx_grp->clone();

    this->assert_equal_to_original_mtx_grp(this->mtx_grp.get());

    set_value_compressed_data<value_type>(
        this->mtx_grp->get_compressed_data(),
        this->mtx_grp->get_num_bytes() - sizeof(value_type), 5.0);
    this->assert_equal_to_original_mtx_grp(dynamic_cast<Mtx*>(clone.get()));
}


TYPED_TEST(Bccoo, CanBeClearedInd)
{
    this->mtx_ind->clear();

    this->assert_empty(this->mtx_ind.get());
}


TYPED_TEST(Bccoo, CanBeClearedGrp)
{
    this->mtx_grp->clear();

    this->assert_empty(this->mtx_grp.get());
}


TYPED_TEST(Bccoo, CanBeReadFromMatrixDataInd)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto m = Mtx::create(this->exec, index_type{BCCOO_BLOCK_SIZE_TESTED},
                         compression::individual);

    m->read({{2, 3},
             {
                 {0, 0, 1.0},
                 {0, 1, 3.0},
                 {0, 2, 2.0},
                 {1, 1, 5.0},
             }});

    this->assert_equal_to_original_mtx_ind(m.get());
}


TYPED_TEST(Bccoo, CanBeReadFromMatrixDataGrp)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto m = Mtx::create(this->exec, index_type{BCCOO_BLOCK_SIZE_TESTED},
                         compression::group);

    m->read({{2, 3},
             {
                 {0, 0, 1.0},
                 {0, 1, 3.0},
                 {0, 2, 2.0},
                 {1, 1, 5.0},
             }});

    this->assert_equal_to_original_mtx_grp(m.get());
}


TYPED_TEST(Bccoo, CanBeReadFromMatrixAssemblyDataInd)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto m = Mtx::create(this->exec, index_type{BCCOO_BLOCK_SIZE_TESTED},
                         compression::individual);
    gko::matrix_assembly_data<value_type, index_type> data(gko::dim<2>{2, 3});
    data.set_value(0, 0, 1.0);
    data.set_value(0, 1, 3.0);
    data.set_value(0, 2, 2.0);
    data.set_value(1, 1, 5.0);

    m->read(data);

    this->assert_equal_to_original_mtx_ind(m.get());
}


TYPED_TEST(Bccoo, CanBeReadFromMatrixAssemblyDataGrp)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto m = Mtx::create(this->exec, index_type{BCCOO_BLOCK_SIZE_TESTED},
                         compression::group);
    gko::matrix_assembly_data<value_type, index_type> data(gko::dim<2>{2, 3});
    data.set_value(0, 0, 1.0);
    data.set_value(0, 1, 3.0);
    data.set_value(0, 2, 2.0);
    data.set_value(1, 1, 5.0);

    m->read(data);

    this->assert_equal_to_original_mtx_grp(m.get());
}


TYPED_TEST(Bccoo, GeneratesCorrectMatrixDataInd)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using tpl = typename gko::matrix_data<value_type, index_type>::nonzero_type;
    gko::matrix_data<value_type, index_type> data;

    this->mtx_ind->write(data);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    ASSERT_EQ(data.nonzeros.size(), 4);
    EXPECT_EQ(data.nonzeros[0], tpl(0, 0, value_type{1.0}));
    EXPECT_EQ(data.nonzeros[1], tpl(0, 1, value_type{3.0}));
    EXPECT_EQ(data.nonzeros[2], tpl(0, 2, value_type{2.0}));
    EXPECT_EQ(data.nonzeros[3], tpl(1, 1, value_type{5.0}));
}


TYPED_TEST(Bccoo, GeneratesCorrectMatrixDataGrp)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using tpl = typename gko::matrix_data<value_type, index_type>::nonzero_type;
    gko::matrix_data<value_type, index_type> data;

    this->mtx_grp->write(data);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    ASSERT_EQ(data.nonzeros.size(), 4);
    EXPECT_EQ(data.nonzeros[0], tpl(0, 0, value_type{1.0}));
    EXPECT_EQ(data.nonzeros[1], tpl(0, 1, value_type{3.0}));
    EXPECT_EQ(data.nonzeros[2], tpl(0, 2, value_type{2.0}));
    EXPECT_EQ(data.nonzeros[3], tpl(1, 1, value_type{5.0}));
}


}  // namespace
