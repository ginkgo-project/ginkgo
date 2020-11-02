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

#ifndef GKO_BENCHMARK_UTILS_INTEL_LINOPS_DP_HPP_
#define GKO_BENCHMARK_UTILS_INTEL_LINOPS_DP_HPP_


#include <ginkgo/ginkgo.hpp>


#include <memory>
#include <oneapi/mkl.hpp>


namespace detail {


inline oneapi::mkl::sparse::matrix_handle_t create_mat_handle()
{
    oneapi::mkl::sparse::matrix_handle_t mat_handle;
    oneapi::mkl::sparse::init_matrix_handle(&mat_handle);
    return mat_handle;
}


class OnemklBase : public gko::LinOp {
public:
    oneapi::mkl::sparse::matrix_handle_t get_mat_handle() const
    {
        return this->mat_handle_.get();
    }

    const gko::DpcppExecutor *get_gpu_exec() const { return gpu_exec_.get(); }

protected:
    void apply_impl(const gko::LinOp *, const gko::LinOp *, const gko::LinOp *,
                    gko::LinOp *) const override
    {
        GKO_NOT_IMPLEMENTED;
    }

    void initialize_mat_handle()
    {
        // TODO: weird
        mat_handle_ = handle_manager<oneapi::mkl::sparse::matrix_handle>(
            create_mat_handle(),
            [](oneapi::mkl::sparse::matrix_handle_t mat_handle) {
                oneapi::mkl::sparse::release_matrix_handle(&mat_handle);
            });
    }

    OnemklBase(std::shared_ptr<const gko::Executor> exec,
               const gko::dim<2> &size = gko::dim<2>{})
        : gko::LinOp(exec, size)
    {
        gpu_exec_ = std::dynamic_pointer_cast<const gko::DpcppExecutor>(exec);
        if (gpu_exec_ == nullptr) {
            GKO_NOT_IMPLEMENTED;
        }
        this->initialize_mat_handle();
    }

    ~OnemklBase() = default;

    OnemklBase(const OnemklBase &other) = delete;

    OnemklBase &operator=(const OnemklBase &other)
    {
        if (this != &other) {
            gko::LinOp::operator=(other);
            this->gpu_exec_ = other.gpu_exec_;
            this->initialize_mat_handle();
        }
        return *this;
    }

private:
    std::shared_ptr<const gko::DpcppExecutor> gpu_exec_;
    template <typename T>
    using handle_manager = std::unique_ptr<T, std::function<void(T *)>>;
    handle_manager<oneapi::mkl::sparse::matrix_handle> mat_handle_;
};


template <bool optimized = false, typename ValueType = gko::default_precision,
          typename IndexType = gko::int32>
class OnemklCsr
    : public gko::EnableLinOp<OnemklCsr<optimized, ValueType, IndexType>,
                              OnemklBase>,
      public gko::EnableCreateMethod<
          OnemklCsr<optimized, ValueType, IndexType>>,
      public gko::ReadableFromMatrixData<ValueType, IndexType> {
    friend class gko::EnableCreateMethod<OnemklCsr>;
    friend class gko::EnablePolymorphicObject<OnemklCsr, OnemklBase>;

public:
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    using mat_data = gko::matrix_data<ValueType, IndexType>;
    void read(const mat_data &data) override
    {
        csr_->read(data);
        this->set_size(gko::dim<2>{csr_->get_size()});

        oneapi::mkl::sparse::set_csr_data(
            this->get_mat_handle(), int(this->get_size()[0]),
            int(this->get_size()[1]), oneapi::mkl::index_base::zero,
            csr_->get_row_ptrs(), csr_->get_col_idxs(), csr_->get_values());
        if (optimized) {
            // need the last argument {} to make sure that it uses USM version.
            oneapi::mkl::sparse::optimize_gemv(
                *(this->get_gpu_exec()->get_queue()), trans_,
                this->get_mat_handle(), {});
        }
    }

    gko::size_type get_num_stored_elements() const noexcept
    {
        return csr_->get_num_stored_elements();
    }

protected:
    void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override
    {
        auto dense_b = gko::as<gko::matrix::Dense<ValueType>>(b);
        auto dense_x = gko::as<gko::matrix::Dense<ValueType>>(x);
        auto db = dense_b->get_const_values();
        auto dx = dense_x->get_values();

        oneapi::mkl::sparse::gemv(*(this->get_gpu_exec()->get_queue()), trans_,
                                  gko::one<ValueType>(), this->get_mat_handle(),
                                  const_cast<double *>(db),
                                  gko::zero<ValueType>(), dx);
    }

    OnemklCsr(std::shared_ptr<const gko::Executor> exec,
              const gko::dim<2> &size = gko::dim<2>{})
        : gko::EnableLinOp<OnemklCsr, OnemklBase>(exec, size),
          csr_(std::move(
              Csr::create(exec, std::make_shared<typename Csr::classical>()))),
          trans_(oneapi::mkl::transpose::nontrans)
    {}


private:
    // Contains {alpha, beta}
    oneapi::mkl::transpose trans_;
    std::shared_ptr<Csr> csr_;
};


}  // namespace detail


using onemkl_csr = detail::OnemklCsr<>;

using onemkl_optimized_csr =
    detail::OnemklCsr<true, gko::default_precision, gko::int32>;


#endif  // GKO_BENCHMARK_UTILS_INTEL_LINOPS_DP_HPP_
