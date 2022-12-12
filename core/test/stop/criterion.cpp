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

#include <ginkgo/core/stop/criterion.hpp>


#include <gtest/gtest.h>


namespace {


struct DummyLogger : public gko::log::Logger {
    DummyLogger() : gko::log::Logger(gko::log::Logger::criterion_events_mask) {}

    void on_criterion_check_started(const gko::stop::Criterion* criterion,
                                    const gko::size_type& num_iterations,
                                    const gko::LinOp* residual,
                                    const gko::LinOp* residual_norm,
                                    const gko::LinOp* solution,
                                    const gko::uint8& stopping_id,
                                    const bool& set_finalized) const override
    {
        criterion_check_started++;
    }

    void on_criterion_check_completed(
        const gko::stop::Criterion* criterion,
        const gko::size_type& num_iterations, const gko::LinOp* residual,
        const gko::LinOp* residual_norm, const gko::LinOp* solution,
        const gko::uint8& stopping_id, const bool& set_finalized,
        const gko::array<gko::stopping_status>* status, const bool& one_changed,
        const bool& all_converged) const override
    {
        criterion_check_completed++;
    }


    mutable int criterion_check_started = 0;
    mutable int criterion_check_completed = 0;
};


class DummyCriterion
    : public gko::EnablePolymorphicObject<DummyCriterion,
                                          gko::stop::Criterion> {
    friend class gko::EnablePolymorphicObject<DummyCriterion,
                                              gko::stop::Criterion>;

public:
    explicit DummyCriterion(std::shared_ptr<const gko::Executor> exec)
        : gko::EnablePolymorphicObject<DummyCriterion, gko::stop::Criterion>(
              std::move(exec))
    {}

protected:
    bool check_impl(gko::uint8 stopping_id, bool set_finalized,
                    gko::array<gko::stopping_status>* stop_status,
                    bool* one_changed, const Updater& updater) override
    {
        return true;
    }
};


class Criterion : public ::testing::Test {
protected:
    Criterion()
        : exec{gko::ReferenceExecutor::create()},
          criterion{std::make_unique<DummyCriterion>(exec)},
          stopping_status{exec},
          logger{std::make_shared<DummyLogger>()}
    {
        criterion->add_logger(logger);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<DummyCriterion> criterion;
    gko::array<gko::stopping_status> stopping_status;
    std::shared_ptr<DummyLogger> logger;
};


TEST_F(Criterion, CanLogCheck)
{
    auto before_logger = *logger;
    bool one_changed = false;

    criterion->check(gko::uint8(0), false, &stopping_status, &one_changed,
                     criterion->update());

    ASSERT_EQ(logger->criterion_check_started,
              before_logger.criterion_check_started + 1);
    ASSERT_EQ(logger->criterion_check_completed,
              before_logger.criterion_check_completed + 1);
}


}  // namespace
