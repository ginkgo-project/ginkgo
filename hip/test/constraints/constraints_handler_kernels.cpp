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

#include <algorithm>
#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/constraints/constraints_handler_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"


namespace {
template <typename ValueIndexType>
class ConsKernels : public ::testing::Test {
public:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using dense = gko::matrix::Dense<value_type>;
    using csr = gko::matrix::Csr<value_type, index_type>;

    ConsKernels()
        : ref(gko::ReferenceExecutor::create()),
          hip(gko::HipExecutor::create(0, ref)),
          rand_engine(42),
          size{32, 30},
          idxs{ref, size[0] / 3},
          didxs{hip, size[0] / 3}
    {
        std::vector<index_type> rows(size[0]);
        std::iota(begin(rows), end(rows), 0);
        std::shuffle(begin(rows), end(rows), rand_engine);
        std::copy_n(begin(rows), idxs.get_num_elems(), idxs.get_data());
        didxs = idxs;
    }

    void TearDown()
    {
        if (hip != nullptr) {
            ASSERT_NO_THROW(hip->synchronize());
        }
    }

    template <typename Matrix>
    auto gen_mtx(gko::size_type num_rows, gko::size_type num_cols)
    {
        return gko::test::generate_random_matrix<Matrix>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(
                std::max(static_cast<gko::size_type>(1), num_cols / 2),
                num_cols),
            std::normal_distribution<>(0.0, 1.0), rand_engine, ref);
    }

    void set_up_data()
    {
        arr = gen_mtx<dense>(size[0], 1);
        darr = gko::clone(hip, arr);
        src_arr = gen_mtx<dense>(size[0], 1);
        dsrc_arr = gko::clone(hip, src_arr);
        mtx = gen_mtx<csr>(size[0], size[1]);
        dmtx = gko::clone(hip, mtx);
    }


    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::HipExecutor> hip;

    std::ranlux48 rand_engine;

    gko::dim<2> size;

    gko::Array<index_type> idxs;
    gko::Array<index_type> didxs;

    std::unique_ptr<dense> arr;
    std::unique_ptr<dense> darr;

    std::unique_ptr<dense> src_arr;
    std::unique_ptr<dense> dsrc_arr;

    std::unique_ptr<csr> mtx;
    std::unique_ptr<csr> dmtx;

    value_type one = gko::one<value_type>();
    value_type zero = gko::zero<value_type>();
};

TYPED_TEST_SUITE(ConsKernels, gko::test::ValueIndexTypes);


TYPED_TEST(ConsKernels, FillSubsetIsEquivalentToRef)
{
    this->set_up_data();

    gko::kernels::reference::cons::fill_subset(
        this->ref, this->idxs, this->arr->get_values(), this->zero);
    gko::kernels::hip::cons::fill_subset(this->hip, this->didxs,
                                         this->darr->get_values(), this->zero);

    GKO_ASSERT_MTX_NEAR(this->arr, this->darr, 0);
}

TYPED_TEST(ConsKernels, CopySubsetIsEquivalentToRef)
{
    this->set_up_data();

    gko::kernels::reference::cons::copy_subset(
        this->ref, this->idxs, this->src_arr->get_const_values(),
        this->arr->get_values());
    gko::kernels::hip::cons::copy_subset(this->hip, this->didxs,
                                         this->dsrc_arr->get_const_values(),
                                         this->darr->get_values());

    GKO_ASSERT_MTX_NEAR(this->arr, this->darr, 0);
}

TYPED_TEST(ConsKernels, SetUnitRowSubsetIsEquivalentToRef)
{
    this->set_up_data();

    gko::kernels::reference::cons::set_unit_rows(
        this->ref, this->idxs, this->mtx->get_const_row_ptrs(),
        this->mtx->get_const_col_idxs(), this->mtx->get_values());
    gko::kernels::hip::cons::set_unit_rows(
        this->hip, this->didxs, this->dmtx->get_const_row_ptrs(),
        this->dmtx->get_const_col_idxs(), this->dmtx->get_values());

    GKO_ASSERT_MTX_NEAR(this->mtx, this->dmtx, 0);
}

}  // namespace
