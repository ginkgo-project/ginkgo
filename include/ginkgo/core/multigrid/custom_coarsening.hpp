// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MULTIGRID_CUSTOM_COARSE_HPP_
#define GKO_PUBLIC_CORE_MULTIGRID_CUSTOM_COARSE_HPP_


#include <vector>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/multigrid/multigrid_level.hpp>

namespace gko {
namespace multigrid {


/**
 * CustomCoarsening allows users to set up the entire multigrid hierarchy.
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup MultigridLevel
 * @ingroup Multigrid
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class CustomCoarsening : public EnableLinOp<CustomCoarsening<ValueType>>,
                         public EnableMultigridLevel<ValueType> {
    friend class EnableLinOp<CustomCoarsening>;
    friend class EnablePolymorphicObject<CustomCoarsening, LinOp>;

public:
    using value_type = ValueType;

    /**
     * Returns the system operator (matrix) of the linear system.
     *
     * @return the system operator (matrix)
     */
    std::shared_ptr<const LinOp> get_system_matrix() const
    {
        return system_matrix_;
    }

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        std::shared_ptr<const LinOp> GKO_FACTORY_PARAMETER_SCALAR(restriction,
                                                                  nullptr);

        std::shared_ptr<const LinOp> GKO_FACTORY_PARAMETER_SCALAR(prologation,
                                                                  nullptr);

        std::shared_ptr<const LinOp> GKO_FACTORY_PARAMETER_SCALAR(coarse,
                                                                  nullptr);
    };
    GKO_ENABLE_LIN_OP_FACTORY(CustomCoarsening, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    void apply_impl(const LinOp* b, LinOp* x) const override
    {
        this->get_composition()->apply(b, x);
    }

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override
    {
        this->get_composition()->apply(alpha, b, beta, x);
    }

    explicit CustomCoarsening(std::shared_ptr<const Executor> exec)
        : EnableLinOp<CustomCoarsening>(std::move(exec))
    {}

    explicit CustomCoarsening(const Factory* factory,
                              std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<CustomCoarsening>(factory->get_executor(),
                                        system_matrix->get_size()),
          EnableMultigridLevel<ValueType>(system_matrix),
          parameters_{factory->get_parameters()},
          system_matrix_{system_matrix}
    {
        GKO_ASSERT(parameters_.restriction != nullptr);
        GKO_ASSERT(parameters_.prologation != nullptr);
        GKO_ASSERT(parameters_.coarse != nullptr);
        if (system_matrix_->get_size()[0] != 0) {
            this->set_multigrid_level(parameters_.prologation,
                                      parameters_.coarse,
                                      parameters_.restriction);
        }
    }

private:
    std::shared_ptr<const LinOp> system_matrix_{};
};


}  // namespace multigrid
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MULTIGRID_CUSTOM_COARSE_HPP_
