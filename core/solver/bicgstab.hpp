/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_SOLVER_BICGSTAB_HPP_
#define GKO_CORE_SOLVER_BICGSTAB_HPP_


#include "core/base/exception.hpp"
#include "core/base/exception_helpers.hpp"
#include "core/base/executor.hpp"
#include "core/base/lin_op.hpp"
#include "core/base/logging.hpp"
#include "core/base/types.hpp"
#include "core/matrix/identity.hpp"


namespace gko {
namespace solver {


template <typename>
class BicgstabFactory;


template <typename ValueType = default_precision>
class Bicgstab : public LinOp, public Loggable {
    friend class BicgstabFactory<ValueType>;

public:
    using value_type = ValueType;

    void copy_from(const LinOp *other) override;

    void copy_from(std::unique_ptr<LinOp> other) override;

    void apply(const LinOp *b, LinOp *x) const override;

    void apply(const LinOp *alpha, const LinOp *b, const LinOp *beta,
               LinOp *x) const override;

    std::unique_ptr<LinOp> clone_type() const override;

    void clear() override;

    std::shared_ptr<const LinOp> get_system_matrix() const
    {
        return system_matrix_;
    }

    int get_max_iters() const { return max_iters_; }

    remove_complex<value_type> get_rel_residual_goal() const
    {
        return rel_residual_goal_;
    }

    void set_precond(std::shared_ptr<const LinOp> precond) noexcept
    {
        precond_ = precond;
    }

    std::shared_ptr<const LinOp> get_precond() const noexcept
    {
        return precond_;
    }

protected:
    Bicgstab(std::shared_ptr<const Executor> exec, int max_iters,
             remove_complex<value_type> rel_residual_goal,
             std::shared_ptr<const LinOp> system_matrix)
        : LinOp(exec, system_matrix->get_num_cols(),
                system_matrix->get_num_rows(),
                system_matrix->get_num_rows() * system_matrix->get_num_cols()),
          system_matrix_(std::move(system_matrix)),
          precond_(matrix::Identity::create(
              std::move(exec), this->get_num_rows(), this->get_num_cols())),
          max_iters_(max_iters),
          rel_residual_goal_(rel_residual_goal)
    {}

    static std::unique_ptr<Bicgstab> create(
        std::shared_ptr<const Executor> exec, int max_iters,
        remove_complex<value_type> rel_residual_goal,
        std::shared_ptr<const LinOp> system_matrix)
    {
        return std::unique_ptr<Bicgstab>(
            new Bicgstab(std::move(exec), max_iters, rel_residual_goal,
                         std::move(system_matrix)));
    }

private:
    std::shared_ptr<const LinOp> system_matrix_;
    std::shared_ptr<const LinOp> precond_;
    int max_iters_;
    remove_complex<value_type> rel_residual_goal_;
};


template <typename ValueType = default_precision>
class BicgstabFactory : public LinOpFactory, public Loggable {
public:
    using value_type = ValueType;

    static std::unique_ptr<BicgstabFactory> create(
        std::shared_ptr<const Executor> exec, int max_iters,
        remove_complex<value_type> rel_residual_goal)
    {
        return std::unique_ptr<BicgstabFactory>(
            new BicgstabFactory(std::move(exec), max_iters, rel_residual_goal));
    }

    std::unique_ptr<LinOp> generate(
        std::shared_ptr<const LinOp> base) const override;

    int get_max_iters() const { return max_iters_; }

    remove_complex<value_type> get_rel_residual_goal() const
    {
        return rel_residual_goal_;
    }

    void set_precond(std::shared_ptr<const LinOpFactory> precond_factory)
    {
        precond_factory_ = precond_factory;
    }

protected:
    BicgstabFactory(std::shared_ptr<const Executor> exec, int max_iters,
                    remove_complex<value_type> rel_residual_goal)
        : LinOpFactory(std::move(exec)),
          max_iters_(max_iters),
          rel_residual_goal_(rel_residual_goal),
          precond_factory_(nullptr)
    {}

    int max_iters_;
    remove_complex<value_type> rel_residual_goal_;
    std::shared_ptr<const LinOpFactory> precond_factory_;
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_CORE_SOLVER_BICGSTAB_HPP
