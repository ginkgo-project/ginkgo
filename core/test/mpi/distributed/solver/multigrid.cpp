// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/config.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/multigrid.hpp>
#include <ginkgo/core/stop/iteration.hpp>

#include "core/test/utils.hpp"


namespace {


using value_type = float;
using local_index_type = int;
using global_index_type = int;
using dist_mtx_type =
    gko::experimental::distributed::Matrix<value_type, local_index_type,
                                           global_index_type>;
using dist_vec_type = gko::experimental::distributed::Vector<value_type>;


class DummyLinOp : public gko::EnableLinOp<DummyLinOp>,
                   public gko::EnableCreateMethod<DummyLinOp> {
public:
    DummyLinOp(std::shared_ptr<const gko::Executor> exec,
               gko::dim<2> size = gko::dim<2>{})
        : EnableLinOp<DummyLinOp>(exec, size)
    {}

protected:
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override {}

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, gko::LinOp* x) const override
    {}
};


template <typename ValueType>
class DummyMultigridLevelWithFactory
    : public gko::EnableLinOp<DummyMultigridLevelWithFactory<ValueType>>,
      public gko::multigrid::EnableMultigridLevel<ValueType> {
public:
    DummyMultigridLevelWithFactory(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableLinOp<DummyMultigridLevelWithFactory>(exec)
    {}


    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory){};
    GKO_ENABLE_LIN_OP_FACTORY(DummyMultigridLevelWithFactory, parameters,
                              Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    DummyMultigridLevelWithFactory(const Factory* factory,
                                   std::shared_ptr<const gko::LinOp> op)
        : gko::EnableLinOp<DummyMultigridLevelWithFactory>(
              factory->get_executor(), op->get_size()),
          gko::multigrid::EnableMultigridLevel<ValueType>(op),
          parameters_{factory->get_parameters()},
          op_{op}
    {
        auto exec = this->get_executor();
        auto distributed_op = dynamic_cast<const dist_mtx_type*>(op_.get());
        auto original_n = distributed_op->get_local_matrix()->get_size()[0];
        gko::size_type n = original_n - 1;

        auto comm = distributed_op->get_communicator();
        auto original_size = op_->get_size()[0];
        auto total_size = original_size - comm.size();
        coarse_ =
            dist_mtx_type::create(exec, comm, gko::dim<2>{total_size},
                                  DummyLinOp::create(exec, gko::dim<2>{n}));
        restrict_ = dist_mtx_type::create(
            exec, comm, gko::dim<2>{total_size, original_size},
            DummyLinOp::create(exec, gko::dim<2>{n, original_n}));
        prolong_ = dist_mtx_type::create(
            exec, comm, gko::dim<2>{original_size, total_size},
            DummyLinOp::create(exec, gko::dim<2>{original_n, n}));
        this->set_multigrid_level(prolong_, coarse_, restrict_);
    }

    std::shared_ptr<const gko::LinOp> op_;
    std::shared_ptr<const gko::LinOp> coarse_;
    std::shared_ptr<const gko::LinOp> restrict_;
    std::shared_ptr<const gko::LinOp> prolong_;

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override {}

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, gko::LinOp* x) const override
    {}
};


class Multigrid : public ::testing::Test {
protected:
    using dist_vec_type = gko::experimental::distributed::Vector<value_type>;
    using multigrid = gko::solver::Multigrid;
    using mg_level = DummyMultigridLevelWithFactory<value_type>;


    Multigrid()
        : ref(gko::ReferenceExecutor::create()),
          comm(gko::experimental::mpi::communicator(MPI_COMM_WORLD))
    {}


    std::shared_ptr<const gko::ReferenceExecutor> ref;
    gko::experimental::mpi::communicator comm;
};


TEST_F(Multigrid, ConstructCorrect)
{
    auto mg_factory =
        multigrid::build()
            .with_max_levels(2u)
            .with_mg_level(mg_level::build())
            .with_pre_smoother(nullptr)
            .with_mid_smoother(nullptr)
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .with_min_coarse_rows(1u)
            .with_coarsest_solver(
                gko::solver::Cg<value_type>::build()
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(1u))
                    .on(this->ref))
            .on(this->ref);
    gko::size_type n = 5;
    gko::size_type global_n = n * this->comm.size();
    auto mtx = gko::share(
        dist_mtx_type::create(this->ref, this->comm, gko::dim<2>{global_n},
                              DummyLinOp::create(this->ref, gko::dim<2>{n})));

    auto mg = mg_factory->generate(mtx);

    auto mg_level = mg->get_mg_level_list();
    auto first_n = global_n - this->comm.size();
    auto second_n = first_n - this->comm.size();
    ASSERT_EQ(mg_level.at(0)->get_fine_op()->get_size(), gko::dim<2>(global_n));
    ASSERT_EQ(mg_level.at(0)->get_restrict_op()->get_size(),
              gko::dim<2>(first_n, global_n));
    ASSERT_EQ(mg_level.at(0)->get_prolong_op()->get_size(),
              gko::dim<2>(global_n, first_n));
    ASSERT_EQ(mg_level.at(0)->get_coarse_op()->get_size(),
              gko::dim<2>(first_n));
    ASSERT_EQ(dynamic_cast<const dist_mtx_type*>(
                  mg_level.at(0)->get_coarse_op().get())
                  ->get_local_matrix()
                  ->get_size(),
              gko::dim<2>(n - 1));
    // next mg_level
    ASSERT_EQ(mg_level.at(1)->get_fine_op()->get_size(), gko::dim<2>(first_n));
    ASSERT_EQ(mg_level.at(1)->get_restrict_op()->get_size(),
              gko::dim<2>(second_n, first_n));
    ASSERT_EQ(mg_level.at(1)->get_prolong_op()->get_size(),
              gko::dim<2>(first_n, second_n));
    ASSERT_EQ(mg_level.at(1)->get_coarse_op()->get_size(),
              gko::dim<2>(second_n));
}


TEST_F(Multigrid, ApplyWithoutException)
{
    auto mg_factory =
        multigrid::build()
            .with_max_levels(2u)
            .with_mg_level(mg_level::build())
            .with_pre_smoother(nullptr)
            .with_mid_smoother(nullptr)
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .with_min_coarse_rows(1u)
            .with_coarsest_solver(
                gko::solver::Cg<value_type>::build()
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(1u))
                    .on(this->ref))
            .on(this->ref);
    gko::size_type n = 5;
    gko::size_type global_n = n * this->comm.size();
    auto mtx = gko::share(
        dist_mtx_type::create(this->ref, this->comm, gko::dim<2>{global_n},
                              DummyLinOp::create(this->ref, gko::dim<2>{n})));
    auto b = dist_vec_type::create(this->ref, this->comm,
                                   gko::dim<2>{global_n, 1}, gko::dim<2>{n, 1});
    auto x = dist_vec_type::create(this->ref, this->comm,
                                   gko::dim<2>{global_n, 1}, gko::dim<2>{n, 1});

    auto mg = mg_factory->generate(mtx);

    ASSERT_NO_THROW(mg->apply(b, x));
}


}  // namespace
