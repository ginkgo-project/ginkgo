// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/pmis_kernels.hpp"

#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>
#include <ginkgo/core/multigrid/pmis.hpp>
#include <ginkgo/core/stop/combined.hpp>

#include "core/test/utils.hpp"


template <typename ValueIndexType>
class Pmis : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using SparsityCsr = gko::matrix::SparsityCsr<value_type, index_type>;
    using MgLevel = gko::multigrid::Pmis<value_type, index_type>;
    using VT = value_type;
    using real_type = gko::remove_complex<value_type>;
    Pmis()
        : exec(gko::ReferenceExecutor::create()),
          pmis_factory(MgLevel::build().with_skip_sorting(true).on(exec))
    {}

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::unique_ptr<typename MgLevel::Factory> pmis_factory;
};

TYPED_TEST_SUITE(Pmis, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


// TODO: some copy/move instruction need to be done when the entire setup is
// ready


TYPED_TEST(Pmis, ComputeStrongDepRow)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    // create a csr
    // allocate sparsity_rows array
    // create the sparsity_rows_answer

    // do operation
    // gko::kernels::reference::pmis::compute_strong_dep_row(this->exec, , ,);

    // check the result
}


TYPED_TEST(Pmis, ComputeStrongDep)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
}


TYPED_TEST(Pmis, InitializeWeightAndStatus)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
}


TYPED_TEST(Pmis, Classify)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    // handle usual cases
}


TYPED_TEST(Pmis, ClassifyWhenEqualWeight)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    // handle cases
}


TYPED_TEST(Pmis, Count)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
}
