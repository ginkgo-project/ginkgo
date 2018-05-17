/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/


#include "core/log/ostream.hpp"
#include "core/matrix/dense.hpp"


#include <iomanip>


namespace gko {
namespace log {


namespace {


template <typename ValueType = default_precision>
std::ostream &operator<<(std::ostream &os, const matrix::Dense<ValueType> *mtx)
{
    auto exec = mtx->get_executor();
    auto tmp = gko::matrix::Dense<ValueType>::create(exec->get_master());
    if (exec != exec->get_master()) {
        tmp->copy_from(mtx);
        mtx = tmp.get();
    }
    os << "[" << std::endl;
    for (int i = 0; i < mtx->get_num_rows(); ++i) {
        for (int j = 0; j < mtx->get_num_cols(); ++j) {
            os << '\t' << mtx->at(i, j);
        }
        os << std::endl;
    }
    return os << "]" << std::endl;
}

#define GKO_DECLARE_OP(_type) \
    std::ostream &operator<<(std::ostream &os, const matrix::Dense<_type> *mtx)
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_OP);
#undef GKO_DECLARE_OP

}  // namespace


template <typename ValueType>
void Ostream<ValueType>::on_iteration_complete(
    const size_type num_iterations) const
{
    os_ << prefix << "iteration " << num_iterations << std::endl;
}


template <typename ValueType>
void Ostream<ValueType>::on_apply(const std::string name) const
{
    os_ << prefix << "starting apply function: " << name << std::endl;
}


/* TODO: improve this whenever the criterion class hierarchy MR is merged */
template <typename ValueType>
void Ostream<ValueType>::on_converged(const size_type at_iteration,
                                      const LinOp *residual) const
{
    os_ << prefix << "converged at iteration " << at_iteration << " residual:\n"
        << as<const gko::matrix::Dense<ValueType>>(residual);
}


#define GKO_DECLARE_OSTREAM(_type) class Ostream<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_OSTREAM);
#undef GKO_DECLARE_OSTREAM


}  // namespace log
}  // namespace gko
