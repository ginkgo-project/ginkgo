// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>
#include <type_traits>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/solver/bicg.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>
#include <ginkgo/core/solver/cb_gmres.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/cgs.hpp>
#include <ginkgo/core/solver/chebyshev.hpp>
#include <ginkgo/core/solver/fcg.hpp>
#include <ginkgo/core/solver/gcr.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/idr.hpp>
#include <ginkgo/core/solver/minres.hpp>
#include <ginkgo/core/solver/pipe_cg.hpp>
#include <ginkgo/core/stop/iteration.hpp>

#include "core/test/utils.hpp"


namespace {


// reports whether it was generated from initialized reuse data, which only
// accepts inputs of the size it was initialized with
class ReuseRecorder : public gko::LinOp {
public:
    ReuseRecorder(std::shared_ptr<const gko::Executor> exec) : gko::LinOp(exec)
    {}

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory){};
    class reuse_data_type : public gko::LinOpFactory::ReuseData {
    public:
        bool initialized = false;
        gko::dim<2> size{};
    };
    GKO_ENABLE_LIN_OP_FACTORY_WITH_REUSE(ReuseRecorder, parameters, Factory,
                                         reuse_data_type);
    GKO_ENABLE_BUILD_METHOD(Factory);

    bool reused = false;

protected:
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override {}

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, gko::LinOp* x) const override
    {}

private:
    ReuseRecorder(const Factory* factory,
                  gko::LinOpGenerateComponents components)
        : gko::LinOp(factory->get_executor(),
                     components.system_matrix->get_size()),
          parameters_{factory->get_parameters()}
    {}

    ReuseRecorder(const Factory* factory,
                  gko::LinOpGenerateComponents components,
                  reuse_data_type& reuse_data)
        : gko::LinOp(factory->get_executor(),
                     components.system_matrix->get_size()),
          parameters_{factory->get_parameters()},
          reused{reuse_data.initialized}
    {
        reuse_data.initialized = true;
        reuse_data.size = this->get_size();
    }

    static void check_reuse_consistent(const Factory*, const gko::LinOp* input,
                                       const reuse_data_type& reuse_data)
    {
        if (reuse_data.initialized) {
            GKO_ASSERT_EQUAL_DIMENSIONS(input->get_size(), reuse_data.size);
        }
    }
};


template <typename SolverType>
class PreconditionedReuse : public ::testing::Test {
protected:
    using Solver = SolverType;
    using Mtx = gko::matrix::Dense<typename Solver::value_type>;

    PreconditionedReuse()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>(
              {{2.0, -1.0, 0.0}, {-1.0, 2.0, -1.0}, {0.0, -1.0, 2.0}}, exec)),
          other_size(gko::initialize<Mtx>({{2.0, -1.0}, {-1.0, 2.0}}, exec)),
          factory(Solver::build()
                      .with_criteria(
                          gko::stop::Iteration::build().with_max_iters(1u))
                      .with_preconditioner(ReuseRecorder::build())
                      .on(exec))
    {}

    static std::shared_ptr<const gko::LinOp> preconditioner(
        const gko::LinOp* solver)
    {
        return dynamic_cast<const gko::Preconditionable*>(solver)
            ->get_preconditioner();
    }

    static bool reused(const gko::LinOp* solver)
    {
        return gko::as<ReuseRecorder>(preconditioner(solver))->reused;
    }

    std::shared_ptr<gko::ReferenceExecutor> exec;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<Mtx> other_size;
    std::unique_ptr<typename Solver::Factory> factory;
};

using Solvers =
    ::testing::Types<gko::solver::Bicg<double>, gko::solver::Bicgstab<double>,
                     gko::solver::CbGmres<double>, gko::solver::Cg<double>,
                     gko::solver::Cgs<double>, gko::solver::Chebyshev<double>,
                     gko::solver::Fcg<double>, gko::solver::Gcr<double>,
                     gko::solver::Gmres<double>, gko::solver::Idr<double>,
                     gko::solver::Minres<double>, gko::solver::PipeCg<double>>;

TYPED_TEST_SUITE(PreconditionedReuse, Solvers, TypenameNameGenerator);


TYPED_TEST(PreconditionedReuse, GenerateReuseForwardsReuseDataToPreconditioner)
{
    auto reuse_data = this->factory->create_empty_reuse_data();

    std::unique_ptr<gko::LinOp> first =
        this->factory->generate_reuse(this->mtx, *reuse_data);
    std::unique_ptr<gko::LinOp> second =
        this->factory->generate_reuse(this->mtx, *reuse_data);
    std::unique_ptr<gko::LinOp> plain = this->factory->generate(this->mtx);

    static_assert(
        std::is_same<decltype(this->factory->generate_reuse(this->mtx,
                                                            *reuse_data)),
                     std::unique_ptr<typename TestFixture::Solver>>::value,
        "generate_reuse has to return the concrete solver type");
    ASSERT_FALSE(this->reused(first.get()));
    ASSERT_TRUE(this->reused(second.get()));
    ASSERT_FALSE(this->reused(plain.get()));
}


TYPED_TEST(PreconditionedReuse, ChecksReuseDataOfPreconditioner)
{
    auto reuse_data = this->factory->create_empty_reuse_data();
    ASSERT_NO_THROW(this->factory->check_reuse_consistent(
        this->other_size.get(), *reuse_data));

    this->factory->generate_reuse(this->mtx, *reuse_data);

    ASSERT_THROW(this->factory->check_reuse_consistent(this->other_size.get(),
                                                       *reuse_data),
                 gko::DimensionMismatch);
    ASSERT_THROW(this->factory->generate_reuse(this->other_size, *reuse_data),
                 gko::DimensionMismatch);
    ASSERT_TRUE(this->reused(
        this->factory->generate_reuse(this->mtx, *reuse_data).get()));
}


TYPED_TEST(PreconditionedReuse, GenerateReuseWithoutPreconditionerFactory)
{
    using Solver = typename TestFixture::Solver;
    auto generated = gko::share(
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .on(this->exec)
            ->generate(this->mtx));
    auto identity_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .on(this->exec);
    auto generated_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .with_generated_preconditioner(generated)
            .with_preconditioner(ReuseRecorder::build())
            .on(this->exec);
    auto identity_data = identity_factory->create_empty_reuse_data();
    auto generated_data = generated_factory->create_empty_reuse_data();

    for (int i = 0; i < 2; i++) {
        std::unique_ptr<gko::LinOp> identity =
            identity_factory->generate_reuse(this->mtx, *identity_data);
        std::unique_ptr<gko::LinOp> with_generated =
            generated_factory->generate_reuse(this->mtx, *generated_data);

        ASSERT_TRUE(dynamic_cast<const gko::matrix::Identity<double>*>(
            this->preconditioner(identity.get()).get()));
        ASSERT_EQ(this->preconditioner(with_generated.get()), generated);
    }
    ASSERT_NO_THROW(generated_factory->check_reuse_consistent(
        this->other_size.get(), *generated_data));
}


}  // namespace
