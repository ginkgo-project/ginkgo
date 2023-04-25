#ifndef GKO_PUBLIC_CORE_REORDER_AMD_HPP_
#define GKO_PUBLIC_CORE_REORDER_AMD_HPP_


#include <memory>


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/permutation.hpp>


namespace gko {
/**
 * @brief The Reorder namespace.
 *
 * @ingroup reorder
 */
namespace reorder {


/**
 * Computes a Approximate Minimum Degree (AMD) reordering of an input
 * matrix.
 *
 * @tparam IndexType  the type used to store sparsity pattern indices of the
 *                    system matrix
 */
template <typename IndexType = int32>
class Amd : public EnablePolymorphicObject<Amd<IndexType>, LinOpFactory>,
            public EnablePolymorphicAssignment<Amd<IndexType>> {
public:
    struct parameters_type;
    friend class EnablePolymorphicObject<Amd<IndexType>, LinOpFactory>;
    friend class enable_parameters_type<parameters_type, Amd<IndexType>>;

    using index_type = IndexType;
    using permutation_type = matrix::Permutation<index_type>;

    struct parameters_type
        : public enable_parameters_type<parameters_type, Amd<IndexType>> {
        /**
         * If set to true, compute a symmetric AMD reordering, otherwise
         * compute a column AMD reordering.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(symmetric, true);
    };

    /**
     * @copydoc LinOpFactory::generate
     * @note This function overrides the default LinOpFactory::generate to
     *       return a Permutation instead of a generic LinOp, which would
     *       need to be cast to Permutation again to access its indices.
     *       It is only necessary because smart pointers aren't covariant.
     */
    std::unique_ptr<permutation_type> generate(
        std::shared_ptr<const LinOp> system_matrix) const;

    /** Creates a new parameter_type to set up the factory. */
    static parameters_type build() { return {}; }

protected:
    explicit Amd(std::shared_ptr<const Executor> exec,
                 const parameters_type& params = {});

    std::unique_ptr<LinOp> generate_impl(
        std::shared_ptr<const LinOp> system_matrix) const override;
};


}  // namespace reorder
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_REORDER_AMD_HPP_