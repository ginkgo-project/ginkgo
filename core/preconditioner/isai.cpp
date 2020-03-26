/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include <ginkgo/core/preconditioner/isai.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/preconditioner/isai_kernels.hpp"


namespace gko {
namespace preconditioner {
namespace isai {


GKO_REGISTER_OPERATION(generate_l, isai::generate_l);
GKO_REGISTER_OPERATION(generate_u, isai::generate_u);


}  // namespace isai


template <typename ValueType, typename IndexType>
std::shared_ptr<LinOp> Isai<ValueType, IndexType>::generate_l(
    const LinOp *to_invert_l)
{
    using Csr = matrix::Csr<ValueType, IndexType>;
    auto exec = this->get_executor();
    auto csr_l = copy_and_convert_to<Csr>(exec, to_invert_l);
    const auto num_elems = csr_l->get_num_stored_elements();

    std::shared_ptr<Csr> inverted_l =
        Csr::create(exec, csr_l->get_size(), num_elems, csr_l->get_strategy());
    exec->run(isai::make_generate_l(csr_l.get(), inverted_l.get()));

    // call make_srow
    inverted_l->set_strategy(inverted_l->get_strategy());
    return {std::move(inverted_l)};
}


template <typename ValueType, typename IndexType>
std::shared_ptr<LinOp> Isai<ValueType, IndexType>::generate_u(
    const LinOp *to_invert_u)
{
    using Csr = matrix::Csr<ValueType, IndexType>;
    auto exec = this->get_executor();
    auto csr_u = copy_and_convert_to<Csr>(exec, to_invert_u);
    const auto num_elems = csr_u->get_num_stored_elements();

    std::shared_ptr<Csr> inverted_u =
        Csr::create(exec, csr_u->get_size(), num_elems, csr_u->get_strategy());
    exec->run(isai::make_generate_l(csr_u.get(), inverted_u.get()));
    // call make_srow
    inverted_u->set_strategy(inverted_u->get_strategy());
    return inverted_u;
}


#define GKO_DECLARE_ISAI(ValueType, IndexType) class Isai<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ISAI);


}  // namespace preconditioner
}  // namespace gko
