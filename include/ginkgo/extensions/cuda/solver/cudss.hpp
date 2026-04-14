// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_EXTENSIONS_CUDA_SOLVER_CUDSS_HPP_
#define GKO_PUBLIC_EXTENSIONS_CUDA_SOLVER_CUDSS_HPP_


#include <memory>

#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>


namespace gko {
namespace ext {
namespace cuda {
namespace solver {


/**
 * A direct solver using NVIDIA's cuDSS library.
 *
 * This solver is only supported on the CudaExecutor. It wraps the cuDSS
 * sparse direct solver, performing analysis, factorization, and solve
 * phases. The factorization is computed during construction (generate)
 * and reused across apply calls.
 *
 * The solver is opaque — factorization data is stored internally in
 * cuDSS-native format and cannot be extracted.
 *
 * @tparam ValueType  the value type of the system matrix and vectors
 * @tparam IndexType  the index type of the system matrix
 */
template <typename ValueType, typename IndexType = int32>
class CuDss : public EnableLinOp<CuDss<ValueType, IndexType>> {
    friend class EnableLinOp<CuDss>;
    friend class EnablePolymorphicObject<CuDss, LinOp>;

public:
    using value_type = ValueType;
    using index_type = IndexType;

    class Factory;

    struct parameters_type : enable_parameters_type<parameters_type, Factory> {
        /**
         * cuDSS matrix type.
         * 0=GENERAL, 1=SYMMETRIC, 2=HERMITIAN, 3=SPD, 4=HPD
         */
        int GKO_FACTORY_PARAMETER_SCALAR(matrix_type, 0);

        /**
         * cuDSS matrix view type.
         * 0=FULL, 1=UPPER, 2=LOWER
         */
        int GKO_FACTORY_PARAMETER_SCALAR(matrix_view, 0);

        /**
         * Reordering algorithm. 0=default.
         */
        int GKO_FACTORY_PARAMETER_SCALAR(reordering_alg, 0);

        /**
         * Enable hybrid host/device execution.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(hybrid_execute, false);

        /**
         * Enable hybrid CPU+GPU memory.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(hybrid_memory, false);
    };
    GKO_ENABLE_LIN_OP_FACTORY(CuDss, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    /**
     * Parse parameters from a configuration property tree.
     */
    static parameters_type parse(
        const config::pnode& config, const config::registry& context,
        const config::type_descriptor& td_for_child =
            config::make_type_descriptor<ValueType, IndexType>());

    /**
     * Returns a configuration_map for registering this type with a
     * config::registry. Users can pass this to the registry constructor
     * to enable JSON/YAML configuration of CuDss.
     */
    static config::configuration_map get_default_config_map();

    /** Creates a copy of the solver (shares factorization state). */
    CuDss(const CuDss&);

    /** Moves from the given solver, leaving it empty. */
    CuDss(CuDss&&) noexcept;

    CuDss& operator=(const CuDss&);

    CuDss& operator=(CuDss&&) noexcept;

    /**
     * Re-run the numeric factorization with updated matrix values.
     *
     * The new matrix must have the same sparsity pattern (dimensions and
     * number of non-zeros) as the matrix used in generate(). Only the
     * numeric factorization phase is re-executed; the symbolic analysis
     * from the initial generate() is reused.
     *
     * @param new_matrix  the updated system matrix (same sparsity pattern)
     */
    void refactorize(std::shared_ptr<const LinOp> new_matrix);

protected:
    explicit CuDss(std::shared_ptr<const Executor> exec);

    CuDss(const Factory* factory, std::shared_ptr<const LinOp> system_matrix);

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

private:
    struct state;
    // system_matrix_ must be declared before state_ so that the CSR data
    // is destroyed after the cuDSS handles that reference it via zero-copy.
    std::shared_ptr<const LinOp> system_matrix_;
    std::shared_ptr<state> state_;
};


}  // namespace solver
}  // namespace cuda
}  // namespace ext
}  // namespace gko


#endif  // GKO_PUBLIC_EXTENSIONS_CUDA_SOLVER_CUDSS_HPP_
