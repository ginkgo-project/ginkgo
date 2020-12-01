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

#ifndef GKO_PUBLIC_CORE_MATRIX_MATRIX_STRATEGIES_HPP_
#define GKO_PUBLIC_CORE_MATRIX_MATRIX_STRATEGIES_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/lin_op.hpp>


namespace gko {
namespace matrix {


namespace matrix_strategy {


template <typename MtxType>
class automatic;

/**
 * strategy_type is to decide how map the work-items to execution units
 *
 * The practical strategy method should inherit strategy_type and implement
 * its `process`, `calc_size` function and the corresponding device kernel.
 */
template <typename MtxType>
class strategy_type {
    friend class automatic<MtxType>;

public:
    using index_type = typename MtxType::index_type;

    /**
     * Creates a strategy_type.
     *
     * @param name  the name of strategy
     */
    strategy_type(std::string name) : name_(name) {}

    /**
     * Returns the name of strategy
     *
     * @return the name of strategy
     */
    std::string get_name() { return name_; }

    /**
     * Computes srow according to row pointers.
     *
     * @param mtx_row_ptrs  the row pointers of the matrix
     * @param mtx_srow  the srow of the matrix
     */
    virtual void process(const Array<index_type> &mtx_row_ptrs,
                         Array<index_type> *mtx_srow) = 0;

    /**
     * Computes the srow size according to the number of nonzeros.
     *
     * @param nnz  the number of nonzeros
     *
     * @return the size of srow
     */
    virtual int64_t calc_size(const int64_t nnz) = 0;

    /**
     * Copy a strategy. This is a workaround until strategies are revamped,
     * since strategies like `automatic` do not work when actually shared.
     */
    virtual std::shared_ptr<strategy_type> copy() = 0;

protected:
    void set_name(std::string name) { name_ = name; }

private:
    std::string name_;
};

/**
 * classical is a strategy_type which uses the same number of threads on
 * each block-row. Classical strategy uses multithreads to calculate on parts of
 * rows and then do a reduction of these threads results. The number of
 * threads per row depends on the max number of stored elements per row.
 */
template <typename MtxType>
class classical : public strategy_type<MtxType> {
public:
    using index_type = typename strategy_type<MtxType>::index_type;

    /**
     * Creates a classical strategy.
     */
    classical() : strategy_type<MtxType>("classical"), max_length_per_row_(0) {}

    void process(const Array<index_type> &mtx_row_ptrs,
                 Array<index_type> *mtx_srow) override
    {
        auto host_mtx_exec = mtx_row_ptrs.get_executor()->get_master();
        Array<index_type> row_ptrs_host(host_mtx_exec);
        const bool is_mtx_on_host{host_mtx_exec == mtx_row_ptrs.get_executor()};
        const index_type *row_ptrs{};
        if (is_mtx_on_host) {
            row_ptrs = mtx_row_ptrs.get_const_data();
        } else {
            row_ptrs_host = mtx_row_ptrs;
            row_ptrs = row_ptrs_host.get_const_data();
        }
        auto num_rows = mtx_row_ptrs.get_num_elems() - 1;
        max_length_per_row_ = 0;
        for (index_type i = 1; i < num_rows + 1; i++) {
            max_length_per_row_ =
                std::max(max_length_per_row_, row_ptrs[i] - row_ptrs[i - 1]);
        }
    }

    int64_t calc_size(const int64_t nnz) override { return 0; }

    index_type get_max_length_per_row() const noexcept
    {
        return max_length_per_row_;
    }

    std::shared_ptr<strategy_type<MtxType>> copy() override
    {
        return std::make_shared<classical<MtxType>>();
    }

private:
    index_type max_length_per_row_;
};

/**
 * load_balance is a strategy_type which uses the load balance algorithm.
 */
template <typename MtxType>
class load_balance : public strategy_type<MtxType> {
public:
    using index_type = typename strategy_type<MtxType>::index_type;

    /**
     * Creates a load_balance strategy.
     */
    load_balance()
        : load_balance(std::move(
              gko::CudaExecutor::create(0, gko::OmpExecutor::create())))
    {}

    /**
     * Creates a load_balance strategy with CUDA executor.
     *
     * @param exec the CUDA executor
     */
    load_balance(std::shared_ptr<const CudaExecutor> exec)
        : load_balance(exec->get_num_warps(), exec->get_warp_size())
    {}

    /**
     * Creates a load_balance strategy with HIP executor.
     *
     * @param exec the HIP executor
     */
    load_balance(std::shared_ptr<const HipExecutor> exec)
        : load_balance(exec->get_num_warps(), exec->get_warp_size(), false)
    {}

    /**
     * Creates a load_balance strategy with specified parameters
     *
     * @param nwarps The number of warps in the executor
     * @param warp_size The warp size of the executor
     * @param cuda_params Whether Nvidia-based warp parameters should be used.
     *
     * @note The warp_size must be the size of full warp. When using this
     *       constructor, set_strategy needs to be called with correct
     *       parameters which is replaced during the conversion.
     */
    load_balance(int64_t nwarps, int warp_size = 32, bool cuda_params = true)
        : strategy_type<MtxType>("load_balance"),
          nwarps_(nwarps),
          warp_size_(warp_size),
          cuda_params_(cuda_params)
    {}

    void process(const Array<index_type> &mtx_row_ptrs,
                 Array<index_type> *mtx_srow) override
    {
        auto nwarps = mtx_srow->get_num_elems();

        if (nwarps > 0) {
            auto host_srow_exec = mtx_srow->get_executor()->get_master();
            auto host_mtx_exec = mtx_row_ptrs.get_executor()->get_master();
            const bool is_srow_on_host{host_srow_exec ==
                                       mtx_srow->get_executor()};
            const bool is_mtx_on_host{host_mtx_exec ==
                                      mtx_row_ptrs.get_executor()};
            Array<index_type> row_ptrs_host(host_mtx_exec);
            Array<index_type> srow_host(host_srow_exec);
            const index_type *row_ptrs{};
            index_type *srow{};
            if (is_srow_on_host) {
                srow = mtx_srow->get_data();
            } else {
                srow_host = *mtx_srow;
                srow = srow_host.get_data();
            }
            if (is_mtx_on_host) {
                row_ptrs = mtx_row_ptrs.get_const_data();
            } else {
                row_ptrs_host = mtx_row_ptrs;
                row_ptrs = row_ptrs_host.get_const_data();
            }
            for (size_type i = 0; i < nwarps; i++) {
                srow[i] = 0;
            }
            const size_type num_rows = mtx_row_ptrs.get_num_elems() - 1;
            const index_type num_elems = row_ptrs[num_rows];
            for (size_type i = 0; i < num_rows; i++) {
                const auto num =
                    (ceildiv(row_ptrs[i + 1], warp_size_) * nwarps);
                const auto den = ceildiv(num_elems, warp_size_);
                auto bucket = ceildiv(num, den);
                if (bucket < nwarps) {
                    srow[bucket]++;
                }
            }
            // find starting row for thread i
            for (size_type i = 1; i < nwarps; i++) {
                srow[i] += srow[i - 1];
            }
            if (!is_srow_on_host) {
                *mtx_srow = srow_host;
            }
        }
    }

    int64_t calc_size(const int64_t nnz) override
    {
        if (warp_size_ > 0) {
            int multiple = 8;
            if (nnz >= 2e8) {
                multiple = 2048;
            } else if (nnz >= 2e7) {
                multiple = 512;
            } else if (nnz >= 2e6) {
                multiple = 128;
            } else if (nnz >= 2e5) {
                multiple = 32;
            }

#if GINKGO_HIP_PLATFORM_HCC
            if (!cuda_params_) {
                multiple = 8;
                if (nnz >= 1e7) {
                    multiple = 64;
                } else if (nnz >= 1e6) {
                    multiple = 16;
                }
            }
#endif  // GINKGO_HIP_PLATFORM_HCC

            auto nwarps = nwarps_ * multiple;
            return min(ceildiv(nnz, warp_size_), int64_t(nwarps));
        } else {
            return 0;
        }
    }

    std::shared_ptr<strategy_type<MtxType>> copy() override
    {
        return std::make_shared<load_balance<MtxType>>(nwarps_, warp_size_,
                                                       cuda_params_);
    }

private:
    int64_t nwarps_;
    int warp_size_;
    bool cuda_params_;
};

template <typename MtxType>
class automatic : public strategy_type<MtxType> {
public:
    using index_type = typename strategy_type<MtxType>::index_type;

    /* Use imbalance strategy when the maximum number of nonzero per row is
     * more than 1024 on NVIDIA hardware */
    const index_type nvidia_row_len_limit = 1024;
    /* Use imbalance strategy when the matrix has more more than 1e6 on
     * NVIDIA hardware */
    const index_type nvidia_nnz_limit = 1e6;
    /* Use imbalance strategy when the maximum number of nonzero per row is
     * more than 768 on AMD hardware */
    const index_type amd_row_len_limit = 768;
    /* Use imbalance strategy when the matrix has more more than 1e8 on AMD
     * hardware */
    const index_type amd_nnz_limit = 1e8;

    /**
     * Creates an automatic strategy.
     */
    automatic()
        : automatic(std::move(
              gko::CudaExecutor::create(0, gko::OmpExecutor::create())))
    {}

    /**
     * Creates an automatic strategy with CUDA executor.
     *
     * @param exec the CUDA executor
     */
    automatic(std::shared_ptr<const CudaExecutor> exec)
        : automatic(exec->get_num_warps(), exec->get_warp_size())
    {}

    /**
     * Creates an automatic strategy with HIP executor.
     *
     * @param exec the HIP executor
     */
    automatic(std::shared_ptr<const HipExecutor> exec)
        : automatic(exec->get_num_warps(), exec->get_warp_size(), false)
    {}

    /**
     * Creates an automatic strategy with specified parameters
     *
     * @param nwarps the number of warps in the executor
     * @param warp_size the warp size of the executor
     * @param cuda_strategy  whether the `cuda_strategy` needs to be used.
     *
     * @note The warp_size must be the size of full warp. When using this
     *       constructor, set_strategy needs to be called with correct
     *       parameters which is replaced during the conversion.
     */
    automatic(int64_t nwarps, int warp_size = 32, bool cuda_strategy = true)
        : strategy_type<MtxType>("automatic"),
          nwarps_(nwarps),
          warp_size_(warp_size),
          cuda_strategy_(cuda_strategy),
          max_length_per_row_(0)
    {}

    void process(const Array<index_type> &mtx_row_ptrs,
                 Array<index_type> *mtx_srow) override
    {
        // if the number of stored elements is larger than <nnz_limit> or
        // the maximum number of stored elements per row is larger than
        // <row_len_limit>, use load_balance otherwise use classical
        index_type nnz_limit = nvidia_nnz_limit;
        index_type row_len_limit = nvidia_row_len_limit;
#if GINKGO_HIP_PLATFORM_HCC
        if (!cuda_strategy_) {
            nnz_limit = amd_nnz_limit;
            row_len_limit = amd_row_len_limit;
        }
#endif  // GINKGO_HIP_PLATFORM_HCC
        auto host_mtx_exec = mtx_row_ptrs.get_executor()->get_master();
        const bool is_mtx_on_host{host_mtx_exec == mtx_row_ptrs.get_executor()};
        Array<index_type> row_ptrs_host(host_mtx_exec);
        const index_type *row_ptrs{};
        if (is_mtx_on_host) {
            row_ptrs = mtx_row_ptrs.get_const_data();
        } else {
            row_ptrs_host = mtx_row_ptrs;
            row_ptrs = row_ptrs_host.get_const_data();
        }
        const auto num_rows = mtx_row_ptrs.get_num_elems() - 1;
        if (row_ptrs[num_rows] > nnz_limit) {
            load_balance<MtxType> actual_strategy(nwarps_, warp_size_,
                                                  cuda_strategy_);
            if (is_mtx_on_host) {
                actual_strategy.process(mtx_row_ptrs, mtx_srow);
            } else {
                actual_strategy.process(row_ptrs_host, mtx_srow);
            }
            this->set_name(actual_strategy.get_name());
        } else {
            index_type maxnum = 0;
            for (index_type i = 1; i < num_rows + 1; i++) {
                maxnum = std::max(maxnum, row_ptrs[i] - row_ptrs[i - 1]);
            }
            if (maxnum > row_len_limit) {
                load_balance<MtxType> actual_strategy(nwarps_, warp_size_,
                                                      cuda_strategy_);
                if (is_mtx_on_host) {
                    actual_strategy.process(mtx_row_ptrs, mtx_srow);
                } else {
                    actual_strategy.process(row_ptrs_host, mtx_srow);
                }
                this->set_name(actual_strategy.get_name());
            } else {
                classical<MtxType> actual_strategy;
                if (is_mtx_on_host) {
                    actual_strategy.process(mtx_row_ptrs, mtx_srow);
                    max_length_per_row_ =
                        actual_strategy.get_max_length_per_row();
                } else {
                    actual_strategy.process(row_ptrs_host, mtx_srow);
                    max_length_per_row_ =
                        actual_strategy.get_max_length_per_row();
                }
                this->set_name(actual_strategy.get_name());
            }
        }
    }

    int64_t calc_size(const int64_t nnz) override
    {
        return std::make_shared<load_balance<MtxType>>(nwarps_, warp_size_,
                                                       cuda_strategy_)
            ->calc_size(nnz);
    }

    index_type get_max_length_per_row() const noexcept
    {
        return max_length_per_row_;
    }

    std::shared_ptr<strategy_type<MtxType>> copy() override
    {
        return std::make_shared<automatic<MtxType>>(nwarps_, warp_size_,
                                                    cuda_strategy_);
    }

private:
    int64_t nwarps_;
    int warp_size_;
    bool cuda_strategy_;
    index_type max_length_per_row_;
};


/**
 * When strategy is load_balance or automatic, rebuild the strategy
 * according to executor's property.
 *
 * @param result  the matrix.
 */
template <typename MtxType>
void strategy_rebuild_helper(MtxType *const result)
{
    // TODO (script:fbcsr): change the code imported from matrix/csr if needed
    // using load_balance = typename Fbcsr<ValueType, IndexType>::load_balance;
    // using automatic = typename Fbcsr<ValueType, IndexType>::automatic;
    auto strategy = result->get_strategy();
    auto executor = result->get_executor();
    if (std::dynamic_pointer_cast<load_balance<MtxType>>(strategy)) {
        if (auto exec =
                std::dynamic_pointer_cast<const HipExecutor>(executor)) {
            result->set_strategy(std::make_shared<load_balance<MtxType>>(exec));
        } else if (auto exec = std::dynamic_pointer_cast<const CudaExecutor>(
                       executor)) {
            result->set_strategy(std::make_shared<load_balance<MtxType>>(exec));
        }
    } else if (std::dynamic_pointer_cast<automatic<MtxType>>(strategy)) {
        if (auto exec =
                std::dynamic_pointer_cast<const HipExecutor>(executor)) {
            result->set_strategy(std::make_shared<automatic<MtxType>>(exec));
        } else if (auto exec = std::dynamic_pointer_cast<const CudaExecutor>(
                       executor)) {
            result->set_strategy(std::make_shared<automatic<MtxType>>(exec));
        }
    }
}


}  // namespace matrix_strategy


}  // namespace matrix
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_MATRIX_MATRIX_STRATEGIES_HPP_
