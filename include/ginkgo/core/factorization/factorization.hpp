// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_FACTORIZATION_FACTORIZATION_HPP_
#define GKO_PUBLIC_CORE_FACTORIZATION_FACTORIZATION_HPP_


#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


namespace gko {
namespace experimental {
namespace factorization {


/**
 * Stores how a Factorization is represented internally. Depending on the
 * representation, different functionality may be available in the class.
 */
enum class storage_type {
    /** The factorization is empty (moved-from or default-constructed). */
    empty,
    /**
     * The two factors are stored as a composition L * U or L * D * U
     * where L and U are Csr matrices and D is a Diagonal matrix.
     */
    composition,
    /*
     * The two factors are stored as a single matrix containing L + U - I, where
     * L has an implicit unit diagonal.
     */
    combined_lu,
    /*
     * The factorization L * D * U is stored as L + D + U - 2I, where
     * L and U have implicit unit diagonals.
     */
    combined_ldu,
    /**
     * The factors are stored as a composition L * L^H or L * D * L^H
     * where L and L^H are Csr matrices and D is a Diagonal matrix.
     */
    symm_composition,
    /*
     * The factorization L * L^H is symmetric and stored as a single matrix
     * containing L + L^H - diag(L).
     */
    symm_combined_cholesky,
    /*
     * The factorization is symmetric and stored as a single matrix containing
     * L + D + L^H - 2 * diag(L), where L and L^H have an implicit unit
     * diagonal.
     */
    symm_combined_ldl,
};


/**
 * Represents a generic factorization consisting of two triangular factors
 * (upper and lower) and an optional diagonal scaling matrix.
 * This class is used to represent a wide range of different factorizations to
 * be passed on to direct solvers and other similar operations. The storage_type
 * represents how the individual factors are stored internally: They may be
 * stored as separate matrices or in a single matrix, and be symmetric or
 * unsymmetric, with the diagonal belonging to both factory, a single factor or
 * being a separate scaling factor (Cholesky vs. LDL^H vs. LU vs. LDU).
 *
 * @tparam ValueType  the value type used to store the factorization entries
 * @tparam IndexType  the index type used to represent the sparsity pattern
 */
template <typename ValueType, typename IndexType>
class Factorization : public EnableLinOp<Factorization<ValueType, IndexType>> {
    friend class EnablePolymorphicObject<Factorization, LinOp>;

public:
    using value_type = ValueType;
    using index_type = IndexType;
    using matrix_type = matrix::Csr<ValueType, IndexType>;
    using diag_type = matrix::Diagonal<ValueType>;
    using composition_type = Composition<ValueType>;

    /**
     * Transforms the factorization from a compact representation suitable only
     * for triangular solves to a composition representation that can also be
     * used to access individual factors and multiply with the factorization.
     *
     * @return  a new Factorization object containing this factorization
     *          represented as storage_type::composition.
     */
    std::unique_ptr<Factorization> unpack() const;

    /** Returns the storage type used by this factorization. */
    storage_type get_storage_type() const;

    /**
     * Returns the lower triangular factor of the factorization, if available,
     * nullptr otherwise.
     */
    std::shared_ptr<const matrix_type> get_lower_factor() const;

    /**
     * Returns the diagonal scaling matrix of the factorization, if available,
     * nullptr otherwise.
     */
    std::shared_ptr<const diag_type> get_diagonal() const;

    /**
     * Returns the upper triangular factor of the factorization, if available,
     * nullptr otherwise.
     */
    std::shared_ptr<const matrix_type> get_upper_factor() const;

    /**
     * Returns the matrix storing a compact representation of the factorization,
     * if available, nullptr otherwise.
     */
    std::shared_ptr<const matrix_type> get_combined() const;

    /** Creates a deep copy of the factorization. */
    Factorization(const Factorization&);

    /** Moves from the given factorization, leaving it empty. */
    Factorization(Factorization&&);

    Factorization& operator=(const Factorization&);

    Factorization& operator=(Factorization&&);

    /**
     * Creates a Factorization from an existing composition.
     * @param composition  the composition consisting of 2 or 3 elements.
     * We expect the first entry to be a lower triangular matrix, and the last
     * entry to be an upper triangular matrix. If the composition has 3
     * elements, we expect the middle entry to be a diagonal matrix.
     *
     * @return  a Factorization storing the elements from the Composition.
     */
    static std::unique_ptr<Factorization> create_from_composition(
        std::unique_ptr<composition_type> composition);

    /**
     * Creates a Factorization from an existing symmetric composition.
     * @param composition  the composition consisting of 2 or 3 elements.
     * We expect the first entry to be a lower triangular matrix, and the last
     * entry to be the transpose of the first entry. If the composition has 3
     * elements, we expect the middle entry to be a diagonal matrix.
     *
     * @return  a symmetric Factorization storing the elements from the
     * Composition.
     */
    static std::unique_ptr<Factorization> create_from_symm_composition(
        std::unique_ptr<composition_type> composition);

    /**
     * Creates a Factorization from an existing combined representation of an LU
     * factorization.
     * @param matrix  the composition consisting of 2 or 3 elements.
     * We expect the first entry to be a lower triangular matrix, and the last
     * entry to be the transpose of the first entry. If the composition has 3
     * elements, we expect the middle entry to be a diagonal matrix.
     *
     * @return  a symmetric Factorization storing the elements from the
     * Composition.
     */
    static std::unique_ptr<Factorization> create_from_combined_lu(
        std::unique_ptr<matrix_type> matrix);

    static std::unique_ptr<Factorization> create_from_combined_ldu(
        std::unique_ptr<matrix_type> matrix);

    static std::unique_ptr<Factorization> create_from_combined_cholesky(
        std::unique_ptr<matrix_type> matrix);

    static std::unique_ptr<Factorization> create_from_combined_ldl(
        std::unique_ptr<matrix_type> matrix);

protected:
    explicit Factorization(std::shared_ptr<const Executor> exec);

    Factorization(std::unique_ptr<Composition<ValueType>> factors,
                  storage_type type);

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

private:
    storage_type storage_type_;
    std::unique_ptr<Composition<ValueType>> factors_;
};


}  // namespace factorization
}  // namespace experimental
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_FACTORIZATION_FACTORIZATION_HPP_
