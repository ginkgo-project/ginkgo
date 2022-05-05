/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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
#include "core/test/utils.hpp"


namespace {


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
          mtx_elm(gko::matrix::Bccoo<value_type, index_type>::create(
              exec, gko::dim<2>{2, 3}, index_type{4}, index_type{1},
              index_type{4 * (sizeof(value_type) + sizeof(index_type) + 1)},
              gko::matrix::bccoo::compression::element)),
          mtx_blk(gko::matrix::Bccoo<value_type, index_type>::create(
              exec, gko::dim<2>{2, 3}, index_type{4}, index_type{1},
              index_type{4 * (sizeof(value_type) + sizeof(index_type) + 1)},
              gko::matrix::bccoo::compression::block))
    {
        mtx_elm->read({{2, 3},
                       {{0, 0, 1.0},
                        {0, 1, 3.0},
                        {0, 2, 2.0},
                        {1, 0, 0.0},
                        {1, 1, 5.0},
                        {1, 2, 0.0}}});
        std::cout << "TEST" << std::endl;
        mtx_blk->read({{2, 3},
                       {{0, 0, 1.0},
                        {0, 1, 3.0},
                        {0, 2, 2.0},
                        {1, 0, 0.0},
                        {1, 1, 5.0},
                        {1, 2, 0.0}}});
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx_elm, mtx_blk;

    void assert_equal_to_original_mtx_elm(const Mtx* m)
    {
        auto rows_data = m->get_const_rows();
        auto offsets_data = m->get_const_offsets();
        auto chunk_data = m->get_const_chunk();

        ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_num_stored_elements(), 4);

        gko::size_type block_size = m->get_block_size();

        std::cout << "ROWS AND OFFSETS" << std::endl;
        index_type row = {};
        index_type offset = {};
        for (index_type i = 0; i < m->get_num_blocks(); i++) {
            std::cout << block_size << " - " << i << " - " << rows_data[i]
                      << " - " << row << " - " << offsets_data[i] << " - "
                      << offset << std::endl;
            EXPECT_EQ(rows_data[i], row);
            EXPECT_EQ(offsets_data[i], offset);
            row += (block_size == 1) && (i == 2);
            offset += (1 + sizeof(value_type)) * block_size +
                      (((block_size == 2) || (block_size >= 4)) &&
                       (i + block_size > 2));
        }
        std::cout << block_size << " - " << m->get_num_blocks() << " - "
                  << offsets_data[m->get_num_blocks()] << " - " << offset
                  << std::endl;
        //       	EXPECT_EQ(offsets_data[m->get_num_blocks()], offset);

        index_type ind = {};
        EXPECT_EQ(chunk_data[ind], 0x00);
        ind++;
        EXPECT_EQ(gko::get_value_chunk<value_type>(chunk_data, ind),
                  value_type{1.0});
        ind += sizeof(value_type);

        EXPECT_EQ(chunk_data[ind], 0x01);
        ind++;
        EXPECT_EQ(gko::get_value_chunk<value_type>(chunk_data, ind),
                  value_type{3.0});
        ind += sizeof(value_type);

        if (block_size < 3) {
            EXPECT_EQ(chunk_data[ind], 0x02);
        } else {
            EXPECT_EQ(chunk_data[ind], 0x01);
        }
        ind++;
        EXPECT_EQ(gko::get_value_chunk<value_type>(chunk_data, ind),
                  value_type{2.0});
        ind += sizeof(value_type);

        if ((block_size == 2) || (block_size >= 4)) {
            EXPECT_EQ(chunk_data[ind], 0xFF);
            ind++;
        }

        EXPECT_EQ(chunk_data[ind], 0x01);
        ind++;
        EXPECT_EQ(gko::get_value_chunk<value_type>(chunk_data, ind),
                  value_type{5.0});
        ind += sizeof(value_type);
    }
    /*
        void assert_equal_to_original_mtx_blk(const Mtx* m)
        {
            auto rows_data = m->get_const_rows();
            auto cols_data = m->get_const_cols();
            auto types_data = m->get_const_types();
            auto offsets_data = m->get_const_offsets();
            auto chunk_data = m->get_const_chunk();

            ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
            ASSERT_EQ(m->get_num_stored_elements(), 4);

            gko::size_type block_size = m->get_block_size();

                                    index_type row = { };
                                    index_type col = { };
                                    index_type type = { };
                                    index_type offset = { };
                                    for (index_type i=0; i<m->get_num_blocks();
       i++) { EXPECT_EQ(rows_data[i], row); EXPECT_EQ(rows_data[i], offset); row
       += (block_size == 1) && (i == 2); offset += (1 + sizeof(value_type)) *
       block_size +
                                                                                                    (((block_size == 2) || (block_size >= 4)) &&
                                                                                                            (i + block_size > 2));
                                    }

            auto chunk_data = m->get_const_chunk();
            gko::size_type block_size = m->get_block_size();
            index_type ind = {};

            ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
            ASSERT_EQ(m->get_num_stored_elements(), 4);

                                    switch (block_size) {
                                    case 1:
                                            break;
                                    case 2:
                                            break;
                                    case 3:
                                            break;
                                    default:
                                            break;
                                    }



            EXPECT_EQ(chunk_data[ind], 0x00);
            ind++;
            EXPECT_EQ(gko::get_value_chunk<value_type>(chunk_data, ind),
                      value_type{1.0});
            ind += sizeof(value_type);


                    }
    */
    void assert_empty_elm(const Mtx* m)
    {
        ASSERT_EQ(m->get_size(), gko::dim<2>(0, 0));
        ASSERT_EQ(m->get_num_stored_elements(), 0);
        ASSERT_EQ(m->get_block_size(), 0);
        ASSERT_EQ(m->get_num_blocks(), 0);
        ASSERT_EQ(m->get_num_bytes(), 0);
        ASSERT_EQ(m->get_const_rows(), nullptr);
        ASSERT_EQ(m->get_const_offsets(), nullptr);
        ASSERT_EQ(m->get_const_chunk(), nullptr);
    }
};

TYPED_TEST_SUITE(Bccoo, gko::test::ValueIndexTypes);


TYPED_TEST(Bccoo, KnowsItsSize)
{
    ASSERT_EQ(this->mtx_elm->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(this->mtx_elm->get_num_stored_elements(), 4);
}


TYPED_TEST(Bccoo, ContainsCorrectData)
{
    this->assert_equal_to_original_mtx_elm(this->mtx_elm.get());
}


TYPED_TEST(Bccoo, CanBeEmpty)
{
    using Mtx = typename TestFixture::Mtx;
    auto mtx_elm = Mtx::create(this->exec);

    this->assert_empty_elm(mtx_elm.get());
}


TYPED_TEST(Bccoo, CanBeCreatedFromExistingData)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;

    const index_type block_size = 10;
    const index_type num_bytes = 6 + 4 * sizeof(value_type);
    index_type ind = {};
    gko::uint8 chunk[num_bytes] = {};
    index_type offsets[] = {0, num_bytes};
    index_type rows[] = {0};

    chunk[ind++] = 0x00;
    gko::set_value_chunk<value_type>(chunk, ind, 1.0);
    ind += sizeof(value_type);
    chunk[ind++] = 0x01;
    gko::set_value_chunk<value_type>(chunk, ind, 2.0);
    ind += sizeof(value_type);
    chunk[ind++] = 0xFF;
    chunk[ind++] = 0x01;
    gko::set_value_chunk<value_type>(chunk, ind, 3.0);
    ind += sizeof(value_type);
    chunk[ind++] = 0xFF;
    chunk[ind++] = 0x00;
    gko::set_value_chunk<value_type>(chunk, ind, 4.0);
    ind += sizeof(value_type);

    auto mtx_elm = gko::matrix::Bccoo<value_type, index_type>::create(
        this->exec, gko::dim<2>{3, 2},
        gko::array<gko::uint8>::view(this->exec, num_bytes, chunk),
        gko::array<index_type>::view(this->exec, 2, offsets),
        gko::array<index_type>::view(this->exec, 1, rows), 4, block_size);

    ASSERT_EQ(mtx_elm->get_num_stored_elements(), 4);
    ASSERT_EQ(mtx_elm->get_block_size(), block_size);
    ASSERT_EQ(mtx_elm->get_const_offsets(), offsets);
    ASSERT_EQ(mtx_elm->get_const_rows(), rows);
}


TYPED_TEST(Bccoo, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto copy = Mtx::create(this->exec);

    copy->copy_from(this->mtx_elm.get());

    this->assert_equal_to_original_mtx_elm(this->mtx_elm.get());
    *((value_type*)(this->mtx_elm->get_chunk() + (2 + sizeof(value_type)))) =
        5.0;
    this->assert_equal_to_original_mtx_elm(copy.get());
}


TYPED_TEST(Bccoo, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    auto copy = Mtx::create(this->exec);

    copy->copy_from(std::move(this->mtx_elm));

    this->assert_equal_to_original_mtx_elm(copy.get());
}


TYPED_TEST(Bccoo, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto clone = this->mtx_elm->clone();

    this->assert_equal_to_original_mtx_elm(this->mtx_elm.get());
    *((value_type*)(this->mtx_elm->get_chunk() + (2 + sizeof(value_type)))) =
        5.0;
    this->assert_equal_to_original_mtx_elm(dynamic_cast<Mtx*>(clone.get()));
}


TYPED_TEST(Bccoo, CanBeCleared)
{
    this->mtx_elm->clear();

    this->assert_empty_elm(this->mtx_elm.get());
}


TYPED_TEST(Bccoo, CanBeReadFromMatrixData)
{
    using Mtx = typename TestFixture::Mtx;
    auto m = Mtx::create(this->exec);
    m->read({{2, 3},
             {{0, 0, 1.0},
              {0, 1, 3.0},
              {0, 2, 2.0},
              {1, 0, 0.0},
              {1, 1, 5.0},
              {1, 2, 0.0}}});

    this->assert_equal_to_original_mtx_elm(m.get());
}


TYPED_TEST(Bccoo, CanBeReadFromMatrixAssemblyData)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto m = Mtx::create(this->exec);
    gko::matrix_assembly_data<value_type, index_type> data(gko::dim<2>{2, 3});
    data.set_value(0, 0, 1.0);
    data.set_value(0, 1, 3.0);
    data.set_value(0, 2, 2.0);
    data.set_value(1, 0, 0.0);
    data.set_value(1, 1, 5.0);
    data.set_value(1, 2, 0.0);

    m->read(data);

    this->assert_equal_to_original_mtx_elm(m.get());
}


TYPED_TEST(Bccoo, GeneratesCorrectMatrixData)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using tpl = typename gko::matrix_data<value_type, index_type>::nonzero_type;
    gko::matrix_data<value_type, index_type> data;

    this->mtx_elm->write(data);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    ASSERT_EQ(data.nonzeros.size(), 4);
    EXPECT_EQ(data.nonzeros[0], tpl(0, 0, value_type{1.0}));
    EXPECT_EQ(data.nonzeros[1], tpl(0, 1, value_type{3.0}));
    EXPECT_EQ(data.nonzeros[2], tpl(0, 2, value_type{2.0}));
    EXPECT_EQ(data.nonzeros[3], tpl(1, 1, value_type{5.0}));
}


}  // namespace
