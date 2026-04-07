// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <optional>
#include <string>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/triangular.hpp>

#include "core/base/dispatch_helper.hpp"
#include "core/config/config_helper.hpp"
#include "core/config/trisolver_config.hpp"
#include "core/solver/upper_trs_kernels.hpp"


namespace gko {
namespace solver {
namespace upper_trs {
namespace {


GKO_REGISTER_OPERATION(generate, upper_trs::generate);
GKO_REGISTER_OPERATION(should_perform_transpose,
                       upper_trs::should_perform_transpose);
GKO_REGISTER_OPERATION(solve, upper_trs::solve);


}  // anonymous namespace
}  // namespace upper_trs


template <typename ValueType, typename IndexType>
typename UpperTrs<ValueType, IndexType>::parameters_type
UpperTrs<ValueType, IndexType>::parse(
    const config::pnode& config, const config::registry& context,
    const config::type_descriptor& td_for_child)
{
    auto params = UpperTrs<ValueType, IndexType>::build();
    config::config_check_decorator config_check(config);
    config::common_trisolver_parse(params, config_check, context, td_for_child);

    return params;
}


template <typename ValueType, typename IndexType>
UpperTrs<ValueType, IndexType>::UpperTrs(const UpperTrs& other)
    : LinOp(other.get_executor(), dim<2>{}, type_to_precision<ValueType>)
{
    *this = other;
}


template <typename ValueType, typename IndexType>
UpperTrs<ValueType, IndexType>::UpperTrs(UpperTrs&& other)
    : LinOp(other.get_executor(), dim<2>{}, type_to_precision<ValueType>)
{
    *this = std::move(other);
}


template <typename ValueType, typename IndexType>
UpperTrs<ValueType, IndexType>& UpperTrs<ValueType, IndexType>::operator=(
    const UpperTrs& other)
{
    if (this != &other) {
        LinOp::operator=(other);
        EnableSolverBase<UpperTrs, CsrMatrix>::operator=(other);
        this->parameters_ = other.parameters_;
        this->generate();
    }
    return *this;
}


template <typename ValueType, typename IndexType>
UpperTrs<ValueType, IndexType>& UpperTrs<ValueType, IndexType>::operator=(
    UpperTrs&& other)
{
    if (this != &other) {
        LinOp::operator=(std::move(other));
        EnableSolverBase<UpperTrs, CsrMatrix>::operator=(std::move(other));
        this->parameters_ = std::exchange(other.parameters_, parameters_type{});
        if (this->get_executor() == other.get_executor()) {
            this->solve_struct_ = std::exchange(other.solve_struct_, nullptr);
        } else {
            this->generate();
        }
    }
    return *this;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> UpperTrs<ValueType, IndexType>::transpose() const
{
    return transposed_type::build()
        .with_num_rhs(this->parameters_.num_rhs)
        .with_algorithm(this->parameters_.algorithm)
        .on(this->get_executor())
        ->generate(share(this->get_system_matrix()->transpose()));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> UpperTrs<ValueType, IndexType>::conj_transpose() const
{
    return transposed_type::build()
        .with_num_rhs(this->parameters_.num_rhs)
        .with_algorithm(this->parameters_.algorithm)
        .on(this->get_executor())
        ->generate(share(this->get_system_matrix()->conj_transpose()));
}


template <typename ValueType, typename IndexType>
void UpperTrs<ValueType, IndexType>::generate()
{
    if (this->get_system_matrix()) {
        this->get_executor()->run(upper_trs::make_generate(
            this->get_system_matrix().get(), this->solve_struct_,
            this->get_parameters().unit_diagonal, parameters_.algorithm,
            parameters_.num_rhs));
    }
}


template <typename ValueType, typename IndexType>
UpperTrs<ValueType, IndexType>::UpperTrs(std::shared_ptr<const Executor> exec)
    : LinOp(std::move(exec), dim<2>{}, type_to_precision<ValueType>)
{}


template <typename ValueType, typename IndexType>
UpperTrs<ValueType, IndexType>::UpperTrs(
    const Factory* factory, std::shared_ptr<const LinOp> system_matrix)
    : LinOp(factory->get_executor(), gko::transpose(system_matrix->get_size()),
            type_to_precision<ValueType>),
      EnableSolverBase<UpperTrs<ValueType, IndexType>, CsrMatrix>{
          copy_and_convert_to<CsrMatrix>(factory->get_executor(),
                                         system_matrix)},
      parameters_{factory->get_parameters()}
{
    this->generate();
}


static bool needs_transpose(std::shared_ptr<const Executor> exec)
{
    bool result{};
    exec->run(upper_trs::make_should_perform_transpose(result));
    return result;
}


template <typename ValueType, typename IndexType>
void UpperTrs<ValueType, IndexType>::apply_impl(const MultiVector* b,
                                                MultiVector* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    apply_precision_dispatch<ValueType>(
        [this](auto view_b, auto view_x) {
            using Vector = matrix::Dense<ValueType>;
            using ws = workspace_traits<UpperTrs>;
            const auto exec = this->get_executor();
            this->setup_workspace();

            // This kernel checks if a transpose is needed for the multiple rhs
            // case. Currently only the algorithm for HIP needs this
            // transposition due to the limitation in the hipsparse algorithm.
            // The other executors (omp and reference, CUDA) do not use the
            // transpose (trans_x and trans_b) and hence are passed in empty
            // pointers.
            Vector* trans_b{};
            Vector* trans_x{};
            using optional_view = std::optional<matrix::view::dense<ValueType>>;
            if (needs_transpose(exec)) {
                trans_b = this->template create_workspace_op<Vector>(
                    ws::transposed_b, gko::transpose(view_b.size));
                trans_x = this->template create_workspace_op<Vector>(
                    ws::transposed_x, gko::transpose(view_x.size));
            }
            exec->run(upper_trs::make_solve(
                this->get_system_matrix().get(), this->solve_struct_.get(),
                this->get_parameters().unit_diagonal, parameters_.algorithm,
                trans_b ? optional_view{trans_b->get_device_view()}
                        : optional_view{},
                trans_x ? optional_view{trans_x->get_device_view()}
                        : optional_view{},
                view_b, view_x));
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void UpperTrs<ValueType, IndexType>::apply_impl(const MultiVector* alpha,
                                                const MultiVector* b,
                                                const MultiVector* beta,
                                                MultiVector* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    LinOp::apply_impl(alpha, b, beta, x);
}


template <typename ValueType, typename IndexType>
int workspace_traits<UpperTrs<ValueType, IndexType>>::num_arrays(const Solver&)
{
    return 0;
}


template <typename ValueType, typename IndexType>
int workspace_traits<UpperTrs<ValueType, IndexType>>::num_vectors(
    const Solver& solver)
{
    return needs_transpose(solver.get_executor()) ? 2 : 0;
}


template <typename ValueType, typename IndexType>
std::vector<std::string>
workspace_traits<UpperTrs<ValueType, IndexType>>::op_names(const Solver& solver)
{
    return needs_transpose(solver.get_executor()) ? std::vector<std::string>{
        "transposed_b",
        "transposed_x",
    } : std::vector<std::string>{};
}


template <typename ValueType, typename IndexType>
std::vector<std::string>
workspace_traits<UpperTrs<ValueType, IndexType>>::array_names(const Solver&)
{
    return {};
}


template <typename ValueType, typename IndexType>
std::vector<int> workspace_traits<UpperTrs<ValueType, IndexType>>::scalars(
    const Solver&)
{
    return {};
}


template <typename ValueType, typename IndexType>
std::vector<int> workspace_traits<UpperTrs<ValueType, IndexType>>::vectors(
    const Solver& solver)
{
    return needs_transpose(solver.get_executor()) ? std::vector<int>{
        transposed_b,
        transposed_x,
    } : std::vector<int>{};
}


#define GKO_DECLARE_UPPER_TRS(ValueType, IndexType) \
    class UpperTrs<ValueType, IndexType>
#define GKO_DECLARE_UPPER_TRS_TRAITS(ValueType, IndexType) \
    struct workspace_traits<UpperTrs<ValueType, IndexType>>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_UPPER_TRS);
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_UPPER_TRS_TRAITS);


}  // namespace solver
}  // namespace gko
