// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_EXTENSIONS_CUDA_SOLVER_CUDSS_HPP_
#define GKO_PUBLIC_EXTENSIONS_CUDA_SOLVER_CUDSS_HPP_


#include <complex>
#include <memory>
#include <type_traits>

#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>


namespace gko {
namespace ext {
namespace cuda {
namespace solver {


namespace detail {


template <typename T>
struct is_cudss_supported_value_type : std::false_type {};

template <>
struct is_cudss_supported_value_type<float> : std::true_type {};
template <>
struct is_cudss_supported_value_type<double> : std::true_type {};
template <>
struct is_cudss_supported_value_type<std::complex<float>> : std::true_type {};
template <>
struct is_cudss_supported_value_type<std::complex<double>> : std::true_type {};


}  // namespace detail


#define GKO_EXT_CUDSS_ASSERT_SUPPORTED_VALUE_TYPE                        \
    static_assert(                                                       \
        ::gko::ext::cuda::solver::detail::is_cudss_supported_value_type< \
            ValueType>::value,                                           \
        "cuDSS only supports float, double, std::complex<float>, and "   \
        "std::complex<double> value types")


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
class Cudss : public Cudss {
    GKO_EXT_CUDSS_ASSERT_SUPPORTED_VALUE_TYPE;
    GKO_ASSERT_SUPPORTED_INDEX_TYPE;

public:
    using value_type = ValueType;
    using index_type = IndexType;

    class Factory;

    struct parameters_type : enable_parameters_type<parameters_type, Factory> {
        /**
         * cuDSS matrix type, mapping to `cudssMatrixType_t`:
         *   - 0 = GENERAL   (unsymmetric)
         *   - 1 = SYMMETRIC (real symmetric)
         *   - 2 = HERMITIAN (complex Hermitian)
         *   - 3 = SPD       (symmetric positive definite)
         *   - 4 = HPD       (Hermitian positive definite)
         *
         * @warning Ginkgo's ::gko::matrix::Csr stores the full matrix. cuDSS
         *          expects that when `matrix_type` is SYMMETRIC / HERMITIAN /
         *          SPD / HPD, the supplied CSR contains **only** the triangle
         *          indicated by `matrix_view` (plus the diagonal). Passing a
         *          fully-stored symmetric matrix with one of these types is
         *          not the documented input contract and can produce
         *          incorrect results. To use the symmetric factorizations,
         *          extract the upper or lower triangle into a new CSR before
         *          constructing the solver. For a fully-stored matrix, use
         *          GENERAL (0) together with `matrix_view = FULL` (0).
         */
        int GKO_FACTORY_PARAMETER_SCALAR(matrix_type, 0);

        /**
         * cuDSS matrix view, mapping to `cudssMatrixViewType_t`:
         *   - 0 = FULL  (entire matrix stored in CSR)
         *   - 1 = LOWER (only strictly-lower + diagonal stored in CSR)
         *   - 2 = UPPER (only strictly-upper + diagonal stored in CSR)
         *
         * This tells cuDSS what the CSR data actually contains; it is not a
         * filter applied to a fully-stored matrix. Use FULL (0) for
         * unsymmetric matrices and when passing a fully-stored symmetric
         * matrix with `matrix_type = GENERAL`. Use LOWER / UPPER only when
         * the CSR itself stores just that triangle (in combination with
         * `matrix_type` SYMMETRIC / HERMITIAN / SPD / HPD). See the
         * `matrix_type` warning above.
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
    GKO_ENABLE_LIN_OP_FACTORY(Cudss, parameters, Factory);
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
     * to enable JSON/YAML configuration of Cudss.
     */
    static config::configuration_map get_config_map();

    /** Creates a copy of the solver (shares factorization state). */
    Cudss(const Cudss&);

    /** Moves from the given solver, leaving it empty. */
    Cudss(Cudss&&) noexcept;

    Cudss& operator=(const Cudss&);

    Cudss& operator=(Cudss&&) noexcept;

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
    explicit Cudss(std::shared_ptr<const Executor> exec);

    Cudss(const Factory* factory, std::shared_ptr<const LinOp> system_matrix);

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
