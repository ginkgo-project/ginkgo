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

#include "mfem-performance.hpp"

#include <ginkgo/ginkgo.hpp>

namespace mfem {
class GinkgoPreconditionerBase : public Solver {
protected:
    GinkgoPreconditionerBase(std::shared_ptr<const gko::Executor> exec,
                             bool on_device, SparseMatrix &a,
                             bool iter_mode = false)
        : Solver(a.Height(), a.Width(), iter_mode)
    {
        // Create Ginkgo preconditioner and generate for wrapped matrix
        auto memclass = a.GetMemoryClass();
        if (memclass == MemoryClass::HOST) {
            std::cout << "SparseMatrix MemoryClass: HOST" << std::endl;
        } else if (memclass == MemoryClass::DEVICE) {
            std::cout << "SparseMatrix MemoryClass: DEVICE" << std::endl;
        } else {
            std::cout << "SparseMatrix MemoryClass: Unknown..." << std::endl;
        }

        exec_ = std::move(exec);
    }

public:
    std::shared_ptr<const gko::Executor> get_exec() { return this->exec_; }
    const gko::LinOpFactory *get_gko_precond_factory()
    {
        return this->gko_precond_factory_.get();
    }
    gko::LinOp *get_gko_precond() { return this->gko_precond_.get(); }


    virtual void Mult(const Vector &x, Vector &y) const;
    virtual void SetOperator(const Operator &op);

protected:
    std::shared_ptr<const gko::Executor> exec_;
    std::unique_ptr<const gko::LinOpFactory> gko_precond_factory_;
    std::unique_ptr<gko::LinOp> gko_precond_;
};


class GinkgoJacobiPreconditioner : public GinkgoPreconditionerBase {
public:
    GinkgoJacobiPreconditioner(std::shared_ptr<const gko::Executor> exec,
                               bool on_device, SparseMatrix &a,
                               bool iter_mode = false)
        : GinkgoPreconditionerBase(exec, on_device, a, iter_mode)
    {
        using mtx = gko::matrix::Csr<double, int>;
        auto gko_sparse =
            mtx::create(exec, gko::dim<2>(a.Height(), a.Width()),
                        gko::Array<double>::view(exec, a.NumNonZeroElems(),
                                                 a.ReadWriteData(on_device)),
                        gko::Array<int>::view(exec, a.NumNonZeroElems(),
                                              a.ReadWriteJ(on_device)),
                        gko::Array<int>::view(exec, a.Height() + 1,
                                              a.ReadWriteI(on_device)));

        gko_precond_factory_ =
            gko::preconditioner::Jacobi<double, int>::build().on(exec);

        gko_precond_ =
            gko_precond_factory_.get()->generate(gko::give(gko_sparse));
    }
};

class GinkgoIluPreconditioner : public GinkgoPreconditionerBase {
public:
    GinkgoIluPreconditioner(std::shared_ptr<const gko::Executor> exec,
                            bool on_device, SparseMatrix &a,
                            bool iter_mode = false)
        : GinkgoPreconditionerBase(exec, on_device, a, iter_mode)
    {
        using mtx = gko::matrix::Csr<double, int>;
        auto gko_sparse =
            mtx::create(exec, gko::dim<2>(a.Height(), a.Width()),
                        gko::Array<double>::view(exec, a.NumNonZeroElems(),
                                                 a.ReadWriteData(on_device)),
                        gko::Array<int>::view(exec, a.NumNonZeroElems(),
                                              a.ReadWriteJ(on_device)),
                        gko::Array<int>::view(exec, a.Height() + 1,
                                              a.ReadWriteI(on_device)));

        gko_precond_factory_ = gko::preconditioner::Ilu<>::build().on(exec);

        gko_precond_ =
            gko_precond_factory_.get()->generate(gko::give(gko_sparse));
    }
};
}  // namespace mfem
