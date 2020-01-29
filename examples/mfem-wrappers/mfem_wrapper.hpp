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

#include "mfem.hpp"

#include <ginkgo/ginkgo.hpp>

template <typename T>
class mfem_destroy {
public:
    using pointer = T *;

    /**
     * Destroys an MFEM object.  Requires object to have a Destroy() method.
     *
     * @param ptr  pointer to the object being deleted
     */
    void operator()(pointer ptr) const noexcept { ptr->Destroy(); }
};

class MFEMVectorWrapper : public gko::matrix::Dense<double> {
public:
    MFEMVectorWrapper(std::shared_ptr<const gko::Executor> exec,
                      gko::size_type size, mfem::Vector *mfem_vec,
                      bool on_device = true, bool ownership = false)
        : gko::matrix::Dense<double>(
              exec, gko::dim<2>{size, 1},
              gko::Array<double>::view(exec, size,
                                       mfem_vec->ReadWrite(on_device)),
              1)
    {
        if (ownership) {
            using deleter = mfem_destroy<mfem::Vector>;
            mfem_vec_ = std::unique_ptr<mfem::Vector,
                                        std::function<void(mfem::Vector *)>>(
                mfem_vec, deleter{});
        } else {
            using deleter = gko::null_deleter<mfem::Vector>;
            mfem_vec_ = std::unique_ptr<mfem::Vector,
                                        std::function<void(mfem::Vector *)>>(
                mfem_vec, deleter{});
        }
    }

    static std::unique_ptr<MFEMVectorWrapper> create(
        std::shared_ptr<const gko::Executor> exec, gko::size_type size,
        mfem::Vector *mfem_vec, bool on_device = true, bool ownership = false)
    {
        return std::unique_ptr<MFEMVectorWrapper>(
            new MFEMVectorWrapper(exec, size, mfem_vec, on_device, ownership));
    }

    mfem::Vector &get_mfem_vec_ref() { return *(this->mfem_vec_.get()); }
    const mfem::Vector &get_mfem_vec_const_ref() const
    {
        return const_cast<const mfem::Vector &>(*(this->mfem_vec_.get()));
    }

    // Override base Dense class implementation
    virtual std::unique_ptr<gko::matrix::Dense<double>>
    create_with_same_config() const override
    {
        mfem::Vector *mfem_vec = new mfem::Vector(
            this->get_size()[0],
            this->mfem_vec_.get()->GetMemory().GetMemoryType());

        mfem_vec->UseDevice(this->mfem_vec_.get()->UseDevice());

        return MFEMVectorWrapper::create(
            this->get_executor(), this->get_size()[0], mfem_vec,
            this->mfem_vec_.get()->UseDevice(), true);
    }


protected:
    void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override;
    void apply_impl(const gko::LinOp *alpha, const gko::LinOp *b,
                    const gko::LinOp *beta, gko::LinOp *x) const override;

private:
    std::unique_ptr<mfem::Vector, std::function<void(mfem::Vector *)>>
        mfem_vec_;
};

class MFEMOperatorWrapper
    : public gko::EnableLinOp<MFEMOperatorWrapper>,
      public gko::EnableCreateMethod<MFEMOperatorWrapper> {
public:
    MFEMOperatorWrapper(std::shared_ptr<const gko::Executor> exec,
                        gko::size_type size = 0,
                        mfem::OperatorHandle oper = mfem::OperatorHandle())
        : gko::EnableLinOp<MFEMOperatorWrapper>(exec, gko::dim<2>{size}),
          gko::EnableCreateMethod<MFEMOperatorWrapper>()
    {
        this->mfem_oper_ = oper;
    }

protected:
    void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override;
    void apply_impl(const gko::LinOp *alpha, const gko::LinOp *b,
                    const gko::LinOp *beta, gko::LinOp *x) const override;

private:
    std::shared_ptr<const gko::LinOp> system_matrix_{};
    mfem::OperatorHandle mfem_oper_;
};
