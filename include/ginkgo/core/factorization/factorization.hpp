/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

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


enum class storage_type {
    /** the factorization is empty (moved-from or default-constructed) */
    empty,
    /**
     * the two factors are stored as a composition L * U or L * D * U
     * where L and U are Csr matrices and D is a Diagonal matrix
     */
    composition,
    /*
     * the two factors are stored as a single matrix containing L + U - I, where
     * L has an implicit unit diagonal
     */
    combined_lu,
    /*
     * the factorization L * D * U is stored as L + D + U - 2I, where
     * L and U have implicit unit diagonals
     */
    combined_ldu,
    /**
     * the two factors are stored as a composition L * L^H or L * D * L^H
     * where L and L^T are Csr matrices and D is a Diagonal matrix
     */
    symm_composition,
    /*
     * the factorization L * L^H is symmetric and stored as a single matrix
     * containing L + L^H - diag(L)
     */
    symm_combined_cholesky,
    /*
     * the factorization is symmetric and stored as a single matrix containing
     * L + D + L^H - 2 * diag(L), where L and L^H have an implicit unit diagonal
     */
    symm_combined_ldl,
};


enum class status { success, failure };


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

    storage_type get_storage_type() const;

    std::shared_ptr<const matrix_type> get_lower_factor() const;

    std::shared_ptr<const diag_type> get_diagonal() const;

    std::shared_ptr<const matrix_type> get_upper_factor() const;

    std::shared_ptr<const matrix_type> get_combined() const;

    const status get_status() const { return status_; };

    /** Creates a deep copy of the factorization. */
    Factorization(const Factorization&);

    /** Moves from the given factorization, leaving it empty. */
    Factorization(Factorization&&);

    Factorization& operator=(const Factorization&);

    Factorization& operator=(Factorization&&);

    static std::unique_ptr<Factorization> create_from_composition(
        std::unique_ptr<composition_type>);

    static std::unique_ptr<Factorization> create_from_symm_composition(
        std::unique_ptr<composition_type>);

    static std::unique_ptr<Factorization> create_from_combined_lu(
        std::unique_ptr<matrix_type>);

    static std::unique_ptr<Factorization> create_from_combined_lu(
        std::unique_ptr<matrix_type>, status stat);

    static std::unique_ptr<Factorization> create_from_combined_ldu(
        std::unique_ptr<matrix_type>);

    static std::unique_ptr<Factorization> create_from_combined_cholesky(
        std::unique_ptr<matrix_type>);

    static std::unique_ptr<Factorization> create_from_combined_ldl(
        std::unique_ptr<matrix_type>);

protected:
    Factorization(std::shared_ptr<const Executor> exec);

    Factorization(std::unique_ptr<Composition<ValueType>> factors,
                  storage_type type);

    Factorization(std::unique_ptr<Composition<ValueType>> factors,
                  storage_type type, status stat);

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

private:
    storage_type storage_type_;
    status status_;
    std::unique_ptr<Composition<ValueType>> factors_;
};


}  // namespace factorization
}  // namespace experimental
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_FACTORIZATION_FACTORIZATION_HPP_
