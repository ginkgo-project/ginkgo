// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/preconditioner/batch_jacobi.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>


class BatchJacobiFactory : public ::testing::Test {
protected:
    using value_type = double;
    using index_type = gko::int32;
    using batch_jacobi_prec =
        gko::batch::preconditioner::Jacobi<value_type, index_type>;

    BatchJacobiFactory()
        : exec(gko::ReferenceExecutor::create()),
          max_block_size(16u),
          block_pointers(gko::array<index_type>(exec->get_master(), 4))
    {
        block_pointers.get_data()[0] = 0;
        block_pointers.get_data()[1] = 2;
        block_pointers.get_data()[2] = 5;
        block_pointers.get_data()[3] = 9;
    }

    std::shared_ptr<const gko::Executor> exec;
    const gko::uint32 max_block_size;
    gko::array<index_type> block_pointers;
};


TEST_F(BatchJacobiFactory, KnowsItsExecutor)
{
    auto batch_jacobi_factory = batch_jacobi_prec::build().on(this->exec);

    ASSERT_EQ(batch_jacobi_factory->get_executor(), this->exec);
}


TEST_F(BatchJacobiFactory, CanSetMaxBlockSize)
{
    auto batch_jacobi_factory =
        batch_jacobi_prec::build().with_max_block_size(16u).on(this->exec);

    ASSERT_EQ(batch_jacobi_factory->get_parameters().max_block_size,
              this->max_block_size);
}


TEST_F(BatchJacobiFactory, CanSetBlockPointers)
{
    gko::array<index_type> block_ptrs_copy(this->exec->get_master(),
                                           this->block_pointers);

    auto batch_jacobi_factory = batch_jacobi_prec::build()
                                    .with_block_pointers(block_ptrs_copy)
                                    .on(this->exec);

    for (int i = 0; i < this->block_pointers.get_size(); i++) {
        ASSERT_EQ(batch_jacobi_factory->get_parameters()
                      .block_pointers.get_const_data()[i],
                  this->block_pointers.get_const_data()[i]);
    }
}
