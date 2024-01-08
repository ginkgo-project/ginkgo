// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_BENCHMARK_UTILS_OVERHEAD_LINOP_HPP_
#define GKO_BENCHMARK_UTILS_OVERHEAD_LINOP_HPP_


#include <cstdint>
#include <memory>
#include <vector>


#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/stop/criterion.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace overhead {


#define GKO_DECLARE_OVERHEAD_OPERATION_KERNEL(_type, _num)            \
    static volatile std::uintptr_t val_operation_##_num = 0;          \
    template <typename _type>                                         \
    void operation##_num(std::shared_ptr<const DefaultExecutor> exec, \
                         const matrix::Dense<_type>* b,               \
                         matrix::Dense<_type>* x)                     \
    {                                                                 \
        val_operation_##_num = reinterpret_cast<std::uintptr_t>(x);   \
    }


#define GKO_DECLARE_ALL                                                      \
    GKO_DECLARE_OVERHEAD_OPERATION_KERNEL(ValueType, 1)                      \
    GKO_DECLARE_OVERHEAD_OPERATION_KERNEL(ValueType, 2)                      \
    GKO_DECLARE_OVERHEAD_OPERATION_KERNEL(ValueType, 3)                      \
    GKO_DECLARE_OVERHEAD_OPERATION_KERNEL(ValueType, 4)                      \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


}  // namespace overhead


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(overhead, GKO_DECLARE_ALL);


#undef GKO_DECLARE_ALL


}  // namespace kernels


namespace overhead {


GKO_REGISTER_OPERATION(operation1, overhead::operation1);
GKO_REGISTER_OPERATION(operation2, overhead::operation2);
GKO_REGISTER_OPERATION(operation3, overhead::operation3);
GKO_REGISTER_OPERATION(operation4, overhead::operation4);


}  // namespace overhead


template <typename ValueType = default_precision>
class Overhead : public EnableLinOp<Overhead<ValueType>>,
                 public Preconditionable {
    friend class EnableLinOp<Overhead>;
    friend class EnablePolymorphicObject<Overhead, LinOp>;

public:
    class Factory;

    struct parameters_type
        : public gko::solver::
              enable_preconditioned_iterative_solver_factory_parameters<
                  parameters_type, Factory> {};

    GKO_ENABLE_LIN_OP_FACTORY(Overhead, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    void apply_impl(const LinOp* b, LinOp* x) const override
    {
        using Vector = matrix::Dense<ValueType>;

        auto exec = this->get_executor();
        auto dense_b = as<const Vector>(b);
        auto dense_x = as<Vector>(x);

        system_matrix_->apply(dense_b, dense_x);
        get_preconditioner()->apply(dense_b, dense_x);

        exec->run(overhead::make_operation1(dense_b, dense_x));
        exec->run(overhead::make_operation2(dense_b, dense_x));
        exec->run(overhead::make_operation3(dense_b, dense_x));
        exec->run(overhead::make_operation4(dense_b, dense_x));
    }

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override
    {
        auto dense_x = as<matrix::Dense<ValueType>>(x);

        auto x_clone = dense_x->clone();
        this->apply(b, x_clone);
        dense_x->scale(beta);
        dense_x->add_scaled(alpha, x_clone);
    }

    explicit Overhead(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Overhead>(std::move(exec))
    {}

    explicit Overhead(const Factory* factory,
                      std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<Overhead>(factory->get_executor(),
                                transpose(system_matrix->get_size())),
          parameters_{factory->get_parameters()},
          system_matrix_{std::move(system_matrix)}
    {
        if (parameters_.generated_preconditioner) {
            GKO_ASSERT_EQUAL_DIMENSIONS(parameters_.generated_preconditioner,
                                        this);
            set_preconditioner(parameters_.generated_preconditioner);
        } else if (parameters_.preconditioner) {
            set_preconditioner(
                parameters_.preconditioner->generate(system_matrix_));
        } else {
            set_preconditioner(matrix::Identity<ValueType>::create(
                this->get_executor(), this->get_size()));
        }
        stop_criterion_factory_ =
            stop::combine(std::move(parameters_.criteria));
    }

private:
    std::shared_ptr<const LinOp> system_matrix_{};
    std::shared_ptr<const stop::CriterionFactory> stop_criterion_factory_{};
};


}  // namespace gko


#endif  // GKO_BENCHMARK_UTILS_OVERHEAD_LINOP_HPP_
