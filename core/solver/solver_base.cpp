// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/workspace.hpp>
#include <ginkgo/core/solver/solver_base.hpp>


namespace gko {
namespace solver {
namespace detail {


SolverBaseLinOp::SolverBaseLinOp(std::shared_ptr<const Executor> exec)
    : WorkspaceHolder(std::move(exec))
{}


SolverBaseLinOp::SolverBaseLinOp(std::shared_ptr<const Executor> exec,
                                 LinOpGenerateComponents& components)
    : WorkspaceHolder(std::move(exec), components)
{}


SolverBaseLinOp::SolverBaseLinOp(const SolverBaseLinOp& other)
    : WorkspaceHolder(other)
{}


SolverBaseLinOp::SolverBaseLinOp(SolverBaseLinOp&& other) noexcept
    : WorkspaceHolder(std::move(other))
{}


SolverBaseLinOp& SolverBaseLinOp::operator=(const SolverBaseLinOp& other)
{
    WorkspaceHolder::operator=(other);
    return *this;
}


SolverBaseLinOp& SolverBaseLinOp::operator=(SolverBaseLinOp&& other) noexcept
{
    WorkspaceHolder::operator=(std::move(other));
    return *this;
}


}  // namespace detail
}  // namespace solver
}  // namespace gko
