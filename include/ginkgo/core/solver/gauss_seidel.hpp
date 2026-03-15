// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_SOLVER_GAUSS_SEIDEL_HPP_
#define GKO_PUBLIC_CORE_SOLVER_GAUSS_SEIDEL_HPP_


#include <memory>
#include <utility>
#include <vector>

#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/solver/solver_base.hpp>


namespace gko {
namespace solver {


/**
 * A helper for algorithm selection in the triangular solvers.
 * It currently only matters for the Cuda executor as there,
 * we have a choice between the Ginkgo syncfree and cuSPARSE implementations.
 */
enum class gs_algorithm { multicolor, syncfree };


/**
 * FwdGaussSeidel is a solver which solves the system A x = b, using the split
 * (D+L) x^(n+1) = b - U x^n
 * where L is the lower triangular part of A, U is the lower triangular part,
 * and D is the diagonal part of A.
 * It works only for certain matrix types: Ell, AMP with Ell.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indices
 *
 * @ingroup solvers
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class FwdGaussSeidel
    : public EnableLinOp<FwdGaussSeidel<ValueType, IndexType>>,
      public EnableSolverBase<FwdGaussSeidel<ValueType, IndexType>>,
      public EnableIterativeBase<FwdGaussSeidel<ValueType, IndexType>> {
    friend class EnableLinOp<FwdGaussSeidel>;
    friend class EnablePolymorphicObject<FwdGaussSeidel, LinOp>;
    friend class EnableIterativeBase<FwdGaussSeidel>;
    GKO_ASSERT_SUPPORTED_VALUE_AND_INDEX_TYPE;

public:
    using value_type = ValueType;
    using index_type = IndexType;

    /**
     * This iterative solver always uses the data in the output vector x
     * as an initial guess.
     *
     * @return  true
     */
    bool apply_uses_initial_guess() const override { return true; }

    class Factory;

    struct parameters_type
        : enable_iterative_solver_factory_parameters<parameters_type, Factory> {
        /**
         * Select the parallel algorithm to be used.
         */
        gs_algorithm GKO_FACTORY_PARAMETER_SCALAR(algorithm,
                                                  gs_algorithm::multicolor);

        /**
         * A host vector of row pointers into the system matrix that denote
         * the starts and ends of each independent set or color.
         *
         * If this parameter is set to an array of size 2 or more,
         * it is assumed that the system matrix is already ordered in the
         * corresponding independent set ordering.
         *
         * Note that color_ptrs[0] == 0 and color_ptrs[num_colors] == num_rows.
         * If the length of this vector is 2, it means the whole row
         * space is one independent set.
         */
        std::vector<IndexType> GKO_FACTORY_PARAMETER_SCALAR(
            color_ptrs, std::vector<IndexType>());
    };
    GKO_ENABLE_LIN_OP_FACTORY(FwdGaussSeidel, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    /**
     * Copy-assignment. Preserves the executor, shallow-copies
     * the system matrix. If the executors mismatch, clones system matrix onto
     * this executor. Solver analysis information will be regenerated.
     */
    FwdGaussSeidel(const FwdGaussSeidel&);

    /**
     * Move-assignment. Preserves the executor, moves
     * the system matrix. If the executors mismatch, clones system matrix onto
     * this executor and regenerates solver analysis information. Moved-from
     * object is empty (0x0 and nullptr system matrix)
     */
    FwdGaussSeidel(FwdGaussSeidel&&);

    /**
     * Copy-constructor. Preserves the executor,
     * shallow-copies the system matrix. Solver analysis information will be
     * regenerated.
     */
    FwdGaussSeidel& operator=(const FwdGaussSeidel&);

    /**
     * Move-constructor. Preserves the executor, moves
     * the system matrix and solver analysis information. Moved-from
     * object is empty (0x0 and nullptr system matrix)
     */
    FwdGaussSeidel& operator=(FwdGaussSeidel&&);

protected:
    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    explicit FwdGaussSeidel(std::shared_ptr<const Executor> exec)
        : EnableLinOp<FwdGaussSeidel>(std::move(exec))
    {}

    explicit FwdGaussSeidel(const Factory* factory,
                            std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<FwdGaussSeidel>(
              factory->get_executor(),
              gko::transpose(system_matrix->get_size())),
          EnableSolverBase<FwdGaussSeidel<ValueType, IndexType>>{system_matrix},
          EnableIterativeBase<FwdGaussSeidel>{
              stop::combine(factory->get_parameters().criteria)},
          parameters_{factory->get_parameters()},
          color_row_ptrs_{parameters_.color_ptrs}
    {
        if (color_row_ptrs_.size() != 0 && color_row_ptrs_.size() < 2) {
            GKO_INVALID_STATE("Color row pointers array has invalid size!");
        }
    }

    template <typename VectorType>
    void apply_dense_impl(const VectorType* b, VectorType* x) const;

    std::vector<IndexType> color_row_ptrs_;
};


template <typename ValueType, typename IndexType>
struct workspace_traits<FwdGaussSeidel<ValueType, IndexType>> {
    using Solver = FwdGaussSeidel<ValueType, IndexType>;
    // number of vectors used by this workspace
    static int num_vectors(const Solver&);
    // number of arrays used by this workspace
    static int num_arrays(const Solver&);
    // array containing the num_vectors names for the workspace vectors
    static std::vector<std::string> op_names(const Solver&);
    // array containing the num_arrays names for the workspace vectors
    static std::vector<std::string> array_names(const Solver&);
    // array containing all varying scalar vectors (independent of problem size)
    static std::vector<int> scalars(const Solver&);
    // array containing all varying vectors (dependent on problem size)
    static std::vector<int> vectors(const Solver&);

    // stopping status array index
    constexpr static int stop = 0;
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_GAUSS_SEIDEL_HPP_
