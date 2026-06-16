// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MULTIGRID_UNIFORM_COARSENING_HPP_
#define GKO_PUBLIC_CORE_MULTIGRID_UNIFORM_COARSENING_HPP_


#include <vector>

#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/config/type_descriptor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/multigrid/multigrid_level.hpp>

namespace gko {
namespace experimental {
namespace distributed {


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
class Matrix;

template <typename LocalIndexType, typename GlobalIndexType>
class Partition;


}  // namespace distributed
}  // namespace experimental


namespace multigrid {


/**
 * UniformCoarsening is a simple coarse grid generation algorithm. It selects
 * the coarse rows by a constant stride `coarse_skip` over the fine-row index
 * space and builds the coarse system from them. The choice is purely
 * index-based — neither matrix values nor mesh geometry are consulted. Fine
 * row `i` either contributes only when it is itself a selected coarse row
 * (injection-style), or is mapped to its nearest coarse row
 * `floor(i / coarse_skip)` (aggregation-style), depending on the
 * `aggregation` parameter.
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
         * The number of rows to skip for the coarse matrix generation
         */
        int GKO_FACTORY_PARAMETER_SCALAR(coarse_skip, 2);

        /**
         * When set to `true` (the default), every fine row `i` is mapped to
         * its nearest coarse row `floor(i / coarse_skip)` (aggregation-style),
         * so that the Galerkin coarse matrix R·A·P preserves graph
         * connectivity. When `false`, only the selected coarse rows
         * (`i % coarse_skip == 0`) participate (injection-style), which can
         * produce disconnected coarse graphs.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(aggregation, true);

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

    /**
     * Create the parameters from the property_tree.
     * Because this is directly tied to the specific type, the value/index type
     * settings within config are ignored and type_descriptor is only used
     * for children configs.
     *
     * @param config  the property tree for setting
     * @param context  the registry
     * @param td_for_child  the type descriptor for children configs. The
     *                      default uses the value/index type of this class.
     *
     * @return parameters
     */
    static parameters_type parse(
        const config::pnode& config, const config::registry& context,
        const config::type_descriptor& td_for_child =
            config::make_type_descriptor<ValueType, IndexType>());

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


    /**
     * This function generates the local matrix coarsening operators.
     *
     * @return a tuple with prolongation, coarse, and restriction linop
     */
    std::tuple<std::shared_ptr<LinOp>, std::shared_ptr<LinOp>,
               std::shared_ptr<LinOp>>
    generate_local(
        std::shared_ptr<const matrix::Csr<ValueType, IndexType>> local_matrix);

#if GINKGO_BUILD_MPI
    /**
     * Communicates the non-local aggregate indices across MPI ranks.
     *
     * @param matrix  a distributed matrix
     * @param coarse_partition  the coarse partition
     * @param local_agg  the local aggregate indices
     *
     * @return  the aggregates for non-local columns in coarse global indexing
     */
    template <typename GlobalIndexType>
    array<GlobalIndexType> communicate_non_local_agg(
        std::shared_ptr<const experimental::distributed::Matrix<
            ValueType, IndexType, GlobalIndexType>>
            matrix,
        std::shared_ptr<
            experimental::distributed::Partition<IndexType, GlobalIndexType>>
            coarse_partition,
        const array<IndexType>& local_agg);
#endif

private:
    std::shared_ptr<const LinOp> system_matrix_{};
    array<IndexType> coarse_rows_;
};


}  // namespace multigrid
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MULTIGRID_UNIFORM_COARSENING_HPP_
