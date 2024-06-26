// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MULTIGRID_UNIFORM_COARSENING_HPP_
#define GKO_PUBLIC_CORE_MULTIGRID_UNIFORM_COARSENING_HPP_


#include <vector>


#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/multigrid/multigrid_level.hpp>

namespace gko {
namespace multigrid {


struct structured_grid {
    int dim = 1;
    std::tuple<int, int, int> size;

    structured_grid(int d, std::tuple<int, int, int> sz) : dim(d), size(sz){};

    bool is_valid()
    {
        return (std::get<0>(size) > 0) && (std::get<1>(size) > 0) &&
               (std::get<2>(size) > 0);
    }
};


struct coarse_spacing {
    int x = 2;
    int y = 1;
    int z = 1;

    coarse_spacing(int x_, int y_, int z_) : x(x_), y(y_), z(z_){};
};


/**
 * UniformCoarsening is a simple coarse grid generation algorithm. It
 * selects the coarse matrix from the fine matrix by constant jumps that can be
 * specified by the user.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes
 *
 * @ingroup MultigridLevel
 * @ingroup Multigrid
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class UniformCoarsening
    : public EnableLinOp<UniformCoarsening<ValueType, IndexType>>,
      public EnableMultigridLevel<ValueType> {
    friend class EnableLinOp<UniformCoarsening>;
    friend class EnablePolymorphicObject<UniformCoarsening, LinOp>;

public:
    using value_type = ValueType;
    using index_type = IndexType;

    /**
     * Returns the system operator (matrix) of the linear system.
     *
     * @return the system operator (matrix)
     */
    std::shared_ptr<const LinOp> get_system_matrix() const
    {
        return system_matrix_;
    }

    /**
     * Returns the selected coarse rows.
     *
     * @return the selected coarse rows.
     */
    IndexType* get_coarse_rows() noexcept { return coarse_rows_.get_data(); }

    /**
     * @copydoc UniformCoarsening::get_coarse_rows()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const IndexType* get_const_coarse_rows() const noexcept
    {
        return coarse_rows_.get_const_data();
    }


    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * The structured grid. The dimension of the grid and the size of the
         * grid can be specified. See @structured_grid
         *
         * The default used is a 1D grid of the size of the number of rows of
         * the matrix.
         */
        structured_grid GKO_FACTORY_PARAMETER_SCALAR(
            grid, structured_grid(1, {-1, -1, -1}));

        /**
         * The spacing in the x, y and z dimensions, as a tuple
         */
        coarse_spacing GKO_FACTORY_PARAMETER_SCALAR(spacing,
                                                    coarse_spacing(2, 1, 1));

        /**
         * The `system_matrix`, which will be given to this factory, must be
         * sorted (first by row, then by column) in order for the algorithm
         * to work. If it is known that the matrix will be sorted, this
         * parameter can be set to `true` to skip the sorting (therefore,
         * shortening the runtime).
         * However, if it is unknown or if the matrix is known to be not sorted,
         * it must remain `false`, otherwise, this multigrid_level might be
         * incorrect.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(skip_sorting, false);
    };
    GKO_ENABLE_LIN_OP_FACTORY(UniformCoarsening, parameters, Factory);
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

    explicit UniformCoarsening(std::shared_ptr<const Executor> exec)
        : EnableLinOp<UniformCoarsening>(std::move(exec))
    {}

    explicit UniformCoarsening(const Factory* factory,
                               std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<UniformCoarsening>(factory->get_executor(),
                                         system_matrix->get_size()),
          EnableMultigridLevel<ValueType>(system_matrix),
          parameters_{factory->get_parameters()},
          system_matrix_{system_matrix}
    {
        if (system_matrix_->get_size()[0] != 0) {
            // generate on the existing matrix
            this->generate();
        }
    }

    void generate();

private:
    std::shared_ptr<const LinOp> system_matrix_{};
    array<IndexType> coarse_rows_;
};


}  // namespace multigrid
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MULTIGRID_UNIFORM_COARSENING_HPP_
