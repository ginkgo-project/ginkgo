/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#ifndef GKO_CORE_FACTORIZATION_PAR_ILU_HPP_
#define GKO_CORE_FACTORIZATION_PAR_ILU_HPP_


#include <memory>


#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
/**
 * @brief The Factorization namespace.
 *
 * @ingroup factor
 */
namespace factorization {


/**
 * A Factory that generates the incomplete LU-factorization
 *
 * @tparam ValueType  Type of the values in the matrices
 * @tparam IndexType  Index type in the matrices
 *
 * @ingroup factor
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class ParIlu : public Composition<ValueType> {
public:
    using value_type = ValueType;
    using index_type = IndexType;
    using l_matrix_type = matrix::Csr<ValueType, IndexType>;
    using u_matrix_type = matrix::Csr<ValueType, IndexType>;

    const l_matrix_type *get_l_factor() const
    {
        // Can be `static_cast` since the type is guaranteed in this class
        return static_cast<const l_matrix_type *>(
            this->get_operators()[0].get());
    }

    const u_matrix_type *get_u_factor() const
    {
        // Can be `static_cast` since the type is guaranteed in this class
        return static_cast<const u_matrix_type *>(
            this->get_operators()[1].get());
    }

    // Remove the possibility of calling `create`, which was enabled by
    // `Composition`
    template <typename... Args>
    static std::unique_ptr<Composition<ValueType>> create(Args &&... args) =
        delete;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        // TODO: maybe change default parameter to something bigger, or make
        // 0 the default one, which would mean 'Automatic', so the
        // implementation can decide an appropriate number of iterations
        unsigned int GKO_FACTORY_PARAMETER(iterations, 1);

        // TODO: Insert all needed parameters in here
    };
    GKO_ENABLE_LIN_OP_FACTORY(ParIlu, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    explicit ParIlu(const Factory *factory,
                    std::shared_ptr<const LinOp> system_matrix)
        : Composition<ValueType>(factory->get_executor()),
          parameters_{factory->get_parameters()}
    {
        // TODO: Fill in potential parameters here
        generate_l_u(std::move(system_matrix))->move_to(this);
    }

    /**
     * Generates the incomplete LU factors and returns a composition of the
     * lower and the upper factor, which will be of the actual type
     * `l_matrix_type` and `u_matrix_type` respectively.
     *
     * @param system_matrix  the source matrix used to generate the
     *                       factors.
     *                       @note: system_matrix must be convertable to a Csr
                                    Matrix, otherwise, an exception is thrown.
     * @return  A composition, containing the incomplete LU factors for the
     *          given system_matrix
     */
    std::unique_ptr<Composition<ValueType>> generate_l_u(
        std::shared_ptr<const LinOp> system_matrix) const;
};


}  // namespace factorization
}  // namespace gko


#endif  // GKO_CORE_FACTORIZATION_PAR_ILU_HPP_
