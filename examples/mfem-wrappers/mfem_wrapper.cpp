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

#include "mfem_wrapper.hpp"

#include <ginkgo/ginkgo.hpp>

void MFEMVectorWrapper::apply_impl(const gko::LinOp *b, gko::LinOp *x) const {}
void MFEMVectorWrapper::apply_impl(const gko::LinOp *alpha, const gko::LinOp *b,
                                   const gko::LinOp *beta, gko::LinOp *x) const
{}

void MFEMOperatorWrapper::apply_impl(const gko::LinOp *b, gko::LinOp *x) const
{
    // Cast to MFEMVectorWrapper; only accept this type for this impl
    const MFEMVectorWrapper *mfem_b = gko::as<const MFEMVectorWrapper>(b);
    MFEMVectorWrapper *mfem_x = gko::as<MFEMVectorWrapper>(x);

    this->mfem_oper_->Mult(mfem_b->get_mfem_vec_const_ref(),
                           mfem_x->get_mfem_vec_ref());
}
void MFEMOperatorWrapper::apply_impl(const gko::LinOp *alpha,
                                     const gko::LinOp *b,
                                     const gko::LinOp *beta,
                                     gko::LinOp *x) const
{
    // x = alpha * op (b) + beta * x

    // Cast to MFEMVectorWrapper; only accept this type for this impl
    const MFEMVectorWrapper *mfem_b = gko::as<const MFEMVectorWrapper>(b);
    MFEMVectorWrapper *mfem_x = gko::as<MFEMVectorWrapper>(x);

    // Check that alpha and beta are Dense<double> of size (1,1):
    if (alpha->get_size()[0] > 1 || alpha->get_size()[1] > 1) {
        throw gko::BadDimension(
            __FILE__, __LINE__, __func__, "alpha", alpha->get_size()[0],
            alpha->get_size()[1],
            "Expected an object of size [1 x 1] for scaling "
            " in this operator's apply_impl");
    }
    if (beta->get_size()[0] > 1 || beta->get_size()[1] > 1) {
        throw gko::BadDimension(
            __FILE__, __LINE__, __func__, "beta", beta->get_size()[0],
            beta->get_size()[1],
            "Expected an object of size [1 x 1] for scaling "
            " in this operator's apply_impl");
    }

    double alpha_f;
    double beta_f;

    if (mfem_x->get_mfem_vec_ref().GetMemory().GetMemoryType() ==
        mfem::MemoryType::CUDA) {
        cudaMemcpy(
            &alpha_f,
            gko::as<gko::matrix::Dense<double>>(alpha)->get_const_values(),
            sizeof(double), cudaMemcpyDeviceToHost);
    } else if (mfem_x->get_mfem_vec_ref().GetMemory().GetMemoryType() ==
               mfem::MemoryType::HOST) {
        alpha_f = gko::as<gko::matrix::Dense<double>>(alpha)->at(0, 0);
    } else {
        std::cout << "error: can't understand MFEM MemoryType of alpha"
                  << std::endl;
    }
    if (mfem_x->get_mfem_vec_ref().GetMemory().GetMemoryType() ==
        mfem::MemoryType::CUDA) {
        cudaMemcpy(
            &beta_f,
            gko::as<gko::matrix::Dense<double>>(beta)->get_const_values(),
            sizeof(double), cudaMemcpyDeviceToHost);
    } else if (mfem_x->get_mfem_vec_ref().GetMemory().GetMemoryType() ==
               mfem::MemoryType::HOST) {
        beta_f = gko::as<gko::matrix::Dense<double>>(beta)->at(0, 0);
    } else {
        std::cout << "error: can't understand MFEM MemoryType of beta"
                  << std::endl;
    }

    // Scale x by beta
    mfem_x->get_mfem_vec_ref() *= beta_f;

    // Multiply operator with b and store in tmp
    mfem::Vector mfem_tmp =
        mfem::Vector(mfem_x->get_size()[0],
                     mfem_x->get_mfem_vec_ref().GetMemory().GetMemoryType());

    // Set UseDevice flag to match mfem_x (not automatically done through
    // MemoryType)
    mfem_tmp.UseDevice(mfem_x->get_mfem_vec_ref().UseDevice());

    this->mfem_oper_->Mult(mfem_b->get_mfem_vec_const_ref(), mfem_tmp);

    // Scale tmp by alpha and add
    mfem_x->get_mfem_vec_ref().Add(alpha_f, mfem_tmp);

    mfem_tmp.Destroy();
}
