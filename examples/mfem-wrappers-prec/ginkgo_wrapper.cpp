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

#include "ginkgo_wrapper.hpp"

#include <ginkgo/ginkgo.hpp>

namespace mfem {
void GinkgoPreconditionerBase::Mult(const Vector &x, Vector &y) const
{
    // !!!!!!
    // TODO: check iterative_mode
    // choose "x or y" way of handling exec/device mismatch....
    // also for creation of preconditioner object?


    // Create Ginkgo wrapped-vectors
    using vec = gko::matrix::Dense<double>;
    bool on_device = false;
    if (x.GetMemory().GetMemoryType() == MemoryType::CUDA) {
        on_device = true;
    }
    auto gko_x = vec::create(
        exec_, gko::dim<2>{x.Size(), 1},
        gko::Array<double>::view(exec_, x.Size(),
                                 const_cast<double *>(x.Read(on_device))),
        1);

    on_device = false;
    if (y.GetMemory().GetMemoryType() == MemoryType::CUDA) {
        on_device = true;
    }
    std::unique_ptr<gko::matrix::Dense<double>> gko_y;
    if (on_device) {
        gko_y = vec::create(
            exec_, gko::dim<2>{y.Size(), 1},
            gko::Array<double>::view(exec_, y.Size(), y.ReadWrite(on_device)),
            1);
    } else {
        gko_y =
            vec::create(exec_->get_master(), gko::dim<2>{y.Size(), 1},
                        gko::Array<double>::view(exec_->get_master(), y.Size(),
                                                 y.ReadWrite(on_device)),
                        1);
    }

    gko_precond_.get()->apply(gko::lend(gko_x), gko::lend(gko_y));
}

void GinkgoPreconditionerBase::SetOperator(const Operator &op)
{
    // Only accept SparseMatrix for this type (see SparseSmoother::SetOperator)
    SparseMatrix *op_mat =
        const_cast<SparseMatrix *>(dynamic_cast<const SparseMatrix *>(&op));
    if (op_mat == NULL) {
        mfem_error("GinkgoPreconditioner::SetOperator : not a SparseMatrix!");
    }
    height = op_mat->Height();
    width = op_mat->Width();

    // Release current preconditioner
    gko_precond_.release();

    bool on_device = false;
    if (exec_ != exec_->get_master()) {
        on_device = true;
    }


    using mtx = gko::matrix::Csr<double, int>;
    auto gko_sparse =
        mtx::create(exec_, gko::dim<2>(op_mat->Height(), op_mat->Width()),
                    gko::Array<double>::view(exec_, op_mat->NumNonZeroElems(),
                                             op_mat->ReadWriteData(on_device)),
                    gko::Array<int>::view(exec_, op_mat->NumNonZeroElems(),
                                          op_mat->ReadWriteJ(on_device)),
                    gko::Array<int>::view(exec_, op_mat->Height() + 1,
                                          op_mat->ReadWriteI(on_device)));
    gko_precond_ = gko_precond_factory_.get()->generate(gko::give(gko_sparse));
}

}  // namespace mfem
