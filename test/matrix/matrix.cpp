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

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


#if GINKGO_DPCPP_SINGLE_MODE
using matrix_value_type = float;
#else
using matrix_value_type = double;
#endif  // GINKGO_DPCPP_SINGLE_MODE


template <typename MtxType>
struct SimpleMatrixTest {
    using matrix_type = MtxType;

    static bool preserves_zeros() { return true; }

    static std::unique_ptr<matrix_type> create(
        std::shared_ptr<gko::Executor> exec, gko::dim<2> size)
    {
        return matrix_type::create(exec->get_master(), size);
    }

    static void modify_data(
        gko::matrix_data<typename MtxType::value_type,
                         typename MtxType::index_type>& data)
    {}

    static void check_property(const std::unique_ptr<matrix_type>&) {}
};

struct DenseWithDefaultStride
    : SimpleMatrixTest<gko::matrix::Dense<matrix_value_type>> {
    static bool preserves_zeros() { return false; }
};

struct DenseWithCustomStride : DenseWithDefaultStride {
    static std::unique_ptr<matrix_type> create(
        std::shared_ptr<gko::Executor> exec, gko::dim<2> size)
    {
        return matrix_type::create(exec, size, size[0] + 10);
    }

    static void check_property(const std::unique_ptr<matrix_type>& mtx)
    {
        ASSERT_EQ(mtx->get_stride(), mtx->get_size()[0] + 10);
    }
};

struct Coo : SimpleMatrixTest<gko::matrix::Coo<matrix_value_type, int>> {};

struct CsrWithDefaultStrategy
    : SimpleMatrixTest<gko::matrix::Csr<matrix_value_type, int>> {};


#if defined(GKO_COMPILING_CUDA) || defined(GKO_COMPILING_HIP) || \
    defined(GKO_COMPILING_DPCPP)


struct CsrWithClassicalStrategy
    : SimpleMatrixTest<gko::matrix::Csr<matrix_value_type, int>> {
    static std::unique_ptr<matrix_type> create(
        std::shared_ptr<gko::Executor> exec, gko::dim<2> size)
    {
        return matrix_type::create(exec, size, 0,
                                   std::make_shared<matrix_type::classical>());
    }

    static void check_property(const std::unique_ptr<matrix_type>& mtx)
    {
        ASSERT_TRUE(dynamic_cast<const matrix_type::classical*>(
            mtx->get_strategy().get()));
    }
};

struct CsrWithMergePathStrategy
    : SimpleMatrixTest<gko::matrix::Csr<matrix_value_type, int>> {
    static std::unique_ptr<matrix_type> create(
        std::shared_ptr<gko::Executor> exec, gko::dim<2> size)
    {
        return matrix_type::create(exec, size, 0,
                                   std::make_shared<matrix_type::merge_path>());
    }

    static void check_property(const std::unique_ptr<matrix_type>& mtx)
    {
        ASSERT_TRUE(dynamic_cast<const matrix_type::merge_path*>(
            mtx->get_strategy().get()));
    }
};

struct CsrWithSparselibStrategy
    : SimpleMatrixTest<gko::matrix::Csr<matrix_value_type, int>> {
    static std::unique_ptr<matrix_type> create(
        std::shared_ptr<gko::Executor> exec, gko::dim<2> size)
    {
        return matrix_type::create(exec, size, 0,
                                   std::make_shared<matrix_type::sparselib>());
    }

    static void check_property(const std::unique_ptr<matrix_type>& mtx)
    {
        ASSERT_TRUE(dynamic_cast<const matrix_type::sparselib*>(
            mtx->get_strategy().get()));
    }
};

struct CsrWithLoadBalanceStrategy
    : SimpleMatrixTest<gko::matrix::Csr<matrix_value_type, int>> {
    static std::unique_ptr<matrix_type> create(
        std::shared_ptr<gko::Executor> exec, gko::dim<2> size)
    {
        return matrix_type::create(exec, size, 0,
                                   std::make_shared<matrix_type::load_balance>(
                                       gko::EXEC_TYPE::create(0, exec)));
    }

    static void check_property(const std::unique_ptr<matrix_type>& mtx)
    {
        ASSERT_TRUE(dynamic_cast<const matrix_type::load_balance*>(
            mtx->get_strategy().get()));
    }
};

struct CsrWithAutomaticalStrategy
    : SimpleMatrixTest<gko::matrix::Csr<matrix_value_type, int>> {
    static std::unique_ptr<matrix_type> create(
        std::shared_ptr<gko::Executor> exec, gko::dim<2> size)
    {
        return matrix_type::create(exec, size, 0,
                                   std::make_shared<matrix_type::automatical>(
                                       gko::EXEC_TYPE::create(0, exec)));
    }

    static void check_property(const std::unique_ptr<matrix_type>& mtx)
    {
        ASSERT_TRUE(dynamic_cast<const matrix_type::automatical*>(
            mtx->get_strategy().get()));
    }
};


#endif


struct Ell : SimpleMatrixTest<gko::matrix::Ell<matrix_value_type, int>> {};


struct FbcsrBlocksize1
    : SimpleMatrixTest<gko::matrix::Fbcsr<matrix_value_type, int>> {
    static bool preserves_zeros() { return false; }

    static std::unique_ptr<matrix_type> create(
        std::shared_ptr<gko::Executor> exec, gko::dim<2> size)
    {
        return matrix_type::create(exec, size, 0, 1);
    }

    static void check_property(const std::unique_ptr<matrix_type>& mtx)
    {
        ASSERT_EQ(mtx->get_block_size(), 1);
    }
};

struct FbcsrBlocksize2
    : SimpleMatrixTest<gko::matrix::Fbcsr<matrix_value_type, int>> {
    static bool preserves_zeros() { return false; }

    static std::unique_ptr<matrix_type> create(
        std::shared_ptr<gko::Executor> exec, gko::dim<2> size)
    {
        return matrix_type::create(exec, size, 0, 2);
    }

    static void check_property(const std::unique_ptr<matrix_type>& mtx)
    {
        ASSERT_EQ(mtx->get_block_size(), 2);
    }
};


struct SellpDefaultParameters
    : SimpleMatrixTest<gko::matrix::Sellp<matrix_value_type, int>> {
    static void check_property(const std::unique_ptr<matrix_type>& mtx)
    {
        ASSERT_EQ(mtx->get_stride_factor(), 1);
        ASSERT_EQ(mtx->get_slice_size(), 64);
    }
};

struct Sellp32Factor2
    : SimpleMatrixTest<gko::matrix::Sellp<matrix_value_type, int>> {
    static std::unique_ptr<matrix_type> create(
        std::shared_ptr<gko::Executor> exec, gko::dim<2> size)
    {
        return matrix_type::create(exec, size, 32, 2, 0);
    }

    static void check_property(const std::unique_ptr<matrix_type>& mtx)
    {
        ASSERT_EQ(mtx->get_stride_factor(), 2);
        ASSERT_EQ(mtx->get_slice_size(), 32);
    }
};


struct HybridDefaultStrategy
    : SimpleMatrixTest<gko::matrix::Hybrid<matrix_value_type, int>> {};

struct HybridColumnLimitStrategy
    : SimpleMatrixTest<gko::matrix::Hybrid<matrix_value_type, int>> {
    static std::unique_ptr<matrix_type> create(
        std::shared_ptr<gko::Executor> exec, gko::dim<2> size)
    {
        return matrix_type::create(
            exec, size, 0, std::make_shared<matrix_type::column_limit>(10));
    }

    static void check_property(const std::unique_ptr<matrix_type>& mtx)
    {
        auto strategy = dynamic_cast<const matrix_type::column_limit*>(
            mtx->get_strategy().get());
        ASSERT_TRUE(strategy);
        ASSERT_EQ(strategy->get_num_columns(), 10);
    }
};

struct HybridImbalanceLimitStrategy
    : SimpleMatrixTest<gko::matrix::Hybrid<matrix_value_type, int>> {
    static std::unique_ptr<matrix_type> create(
        std::shared_ptr<gko::Executor> exec, gko::dim<2> size)
    {
        return matrix_type::create(
            exec, size, 0, std::make_shared<matrix_type::imbalance_limit>(0.5));
    }

    static void check_property(const std::unique_ptr<matrix_type>& mtx)
    {
        auto strategy = dynamic_cast<const matrix_type::imbalance_limit*>(
            mtx->get_strategy().get());
        ASSERT_TRUE(strategy);
        ASSERT_EQ(strategy->get_percentage(), 0.5);
    }
};

struct HybridImbalanceBoundedLimitStrategy
    : SimpleMatrixTest<gko::matrix::Hybrid<matrix_value_type, int>> {
    static std::unique_ptr<matrix_type> create(
        std::shared_ptr<gko::Executor> exec, gko::dim<2> size)
    {
        return matrix_type::create(
            exec, size, 0,
            std::make_shared<matrix_type::imbalance_bounded_limit>(0.5, 0.01));
    }

    static void check_property(const std::unique_ptr<matrix_type>& mtx)
    {
        auto strategy =
            dynamic_cast<const matrix_type::imbalance_bounded_limit*>(
                mtx->get_strategy().get());
        ASSERT_TRUE(strategy);
        ASSERT_EQ(strategy->get_percentage(), 0.5);
        ASSERT_EQ(strategy->get_ratio(), 0.01);
    }
};

struct HybridMinStorageStrategy
    : SimpleMatrixTest<gko::matrix::Hybrid<matrix_value_type, int>> {
    static std::unique_ptr<matrix_type> create(
        std::shared_ptr<gko::Executor> exec, gko::dim<2> size)
    {
        return matrix_type::create(
            exec, size, 0,
            std::make_shared<matrix_type::minimal_storage_limit>());
    }

    static void check_property(const std::unique_ptr<matrix_type>& mtx)
    {
        auto strategy = dynamic_cast<const matrix_type::minimal_storage_limit*>(
            mtx->get_strategy().get());
        ASSERT_TRUE(strategy);
    }
};

struct HybridAutomaticStrategy
    : SimpleMatrixTest<gko::matrix::Hybrid<matrix_value_type, int>> {
    static std::unique_ptr<matrix_type> create(
        std::shared_ptr<gko::Executor> exec, gko::dim<2> size)
    {
        return matrix_type::create(exec, size, 0,
                                   std::make_shared<matrix_type::automatic>());
    }

    static void check_property(const std::unique_ptr<matrix_type>& mtx)
    {
        auto strategy = dynamic_cast<const matrix_type::automatic*>(
            mtx->get_strategy().get());
        ASSERT_TRUE(strategy);
    }
};


struct SparsityCsr
    : SimpleMatrixTest<gko::matrix::SparsityCsr<matrix_value_type, int>> {
    static void modify_data(gko::matrix_data<matrix_value_type, int>& data)
    {
        using entry_type =
            gko::matrix_data<matrix_value_type, int>::nonzero_type;
        for (auto& entry : data.nonzeros) {
            entry.value = gko::one<matrix_value_type>();
        }
    }
};


template <typename ObjectType>
struct test_pair {
    std::unique_ptr<ObjectType> ref;
    std::unique_ptr<ObjectType> dev;

    test_pair(std::unique_ptr<ObjectType> ref_obj,
              std::shared_ptr<const gko::Executor> exec)
        : ref{std::move(ref_obj)}, dev{gko::clone(exec, ref)}
    {}

    test_pair(std::unique_ptr<ObjectType> ref_obj,
              std::unique_ptr<ObjectType> dev_obj)
        : ref{std::move(ref_obj)}, dev{std::move(dev_obj)}
    {}
};


template <typename T>
class Matrix : public ::testing::Test {
protected:
    using Config = T;
    using Mtx = typename T::matrix_type;
    using index_type = typename Mtx::index_type;
    using value_type = typename Mtx::value_type;
    using mixed_value_type = gko::next_precision<value_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using MixedVec = gko::matrix::Dense<mixed_value_type>;

    Matrix() : rand_engine(15) {}

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        init_executor(ref, exec);
    }

    void TearDown()
    {
        if (exec != nullptr) {
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

    template <typename DistType>
    gko::matrix_data<value_type, index_type> gen_mtx_data(int num_rows,
                                                          int num_cols,
                                                          DistType dist)
    {
        return gko::test::generate_random_matrix_data<value_type, index_type>(
            num_rows, num_cols, dist, std::normal_distribution<>(0.0, 1.0),
            rand_engine);
    }

    gko::matrix_data<value_type, index_type> gen_mtx_data(int num_rows,
                                                          int num_cols,
                                                          int min_cols,
                                                          int max_cols)
    {
        return gen_mtx_data(
            num_rows, num_cols,
            std::uniform_int_distribution<>(min_cols, max_cols));
    }

    template <typename ValueType, typename IndexType>
    gko::matrix_data<ValueType, IndexType> gen_dense_data(gko::dim<2> size)
    {
        return {
            size,
            std::normal_distribution<gko::remove_complex<ValueType>>(0.0, 1.0),
            rand_engine};
    }

    template <typename VecType = Vec>
    test_pair<VecType> gen_in_vec(const test_pair<Mtx>& mtx, int nrhs,
                                  int stride)
    {
        auto size = gko::dim<2>{mtx.ref->get_size()[1],
                                static_cast<gko::size_type>(nrhs)};
        auto result = VecType::create(ref, size, stride);
        result->read(gen_dense_data<typename VecType::value_type,
                                    typename Mtx::index_type>(size));
        return {std::move(result), exec};
    }

    template <typename VecType = Vec>
    test_pair<VecType> gen_scalar()
    {
        return {gko::initialize<VecType>(
                    {gko::test::detail::get_rand_value<
                        typename VecType::value_type>(
                        std::normal_distribution<
                            gko::remove_complex<typename VecType::value_type>>(
                            0.0, 1.0),
                        rand_engine)},
                    ref),
                exec};
    }

    template <typename VecType = Vec>
    test_pair<VecType> gen_out_vec(const test_pair<Mtx>& mtx, int nrhs,
                                   int stride)
    {
        auto size = gko::dim<2>{mtx.ref->get_size()[0],
                                static_cast<gko::size_type>(nrhs)};
        auto result = VecType::create(ref, size, stride);
        result->read(gen_dense_data<typename VecType::value_type,
                                    typename Mtx::index_type>(size));
        return {std::move(result), exec};
    }

    double tol() { return r<value_type>::value; }

    double mixed_tol() { return r_mixed<value_type, mixed_value_type>(); }

    template <typename TestFunction>
    void forall_matrix_data_scenarios(TestFunction fn)
    {
        auto guarded_fn = [&](auto mtx) {
            try {
                fn(std::move(mtx));
            } catch (std::exception& e) {
                FAIL() << e.what();
            }
        };
        {
            SCOPED_TRACE("Zero matrix (0x0)");
            guarded_fn(gen_mtx_data(0, 0, 0, 0));
        }
        {
            SCOPED_TRACE("Zero matrix (0x2)");
            guarded_fn(gen_mtx_data(0, 2, 0, 0));
        }
        {
            SCOPED_TRACE("Zero matrix (2x0)");
            guarded_fn(gen_mtx_data(2, 0, 0, 0));
        }
        {
            SCOPED_TRACE("Zero matrix (200x100)");
            guarded_fn(gen_mtx_data(200, 100, 0, 0));
        }
        {
            SCOPED_TRACE("Sparse Matrix with some zeros rows (200x100)");
            guarded_fn(gen_mtx_data(200, 100, 0, 50));
        }
        {
            SCOPED_TRACE("Sparse Matrix with fixed row nnz (200x100)");
            guarded_fn(gen_mtx_data(200, 100, 50, 50));
        }
        {
            SCOPED_TRACE("Sparse Matrix with variable row nnz (200x100)");
            guarded_fn(gen_mtx_data(200, 100, 10, 50));
        }
        {
            SCOPED_TRACE(
                "Sparse Matrix with heavily imbalanced row nnz (200x100)");
            guarded_fn(
                gen_mtx_data(200, 100, std::poisson_distribution<>{1.5}));
        }
        {
            SCOPED_TRACE("Dense matrix (200x100)");
            guarded_fn(gen_mtx_data(200, 100, 100, 100));
        }
    }

    template <typename TestFunction>
    void forall_matrix_scenarios(TestFunction fn)
    {
        auto guarded_fn = [&](auto mtx) {
            try {
                T::check_property(mtx.ref);
                T::check_property(mtx.dev);
                fn(std::move(mtx));
            } catch (std::exception& e) {
                FAIL() << e.what();
            }
        };
        {
            SCOPED_TRACE("Uninitialized matrix (0x0)");
            guarded_fn(test_pair<Mtx>{T::create(ref, gko::dim<2>{}),
                                      T::create(exec, gko::dim<2>{})});
        }
        {
            SCOPED_TRACE("Uninitialized matrix (0x2)");
            guarded_fn(test_pair<Mtx>{T::create(ref, gko::dim<2>{0, 2}),
                                      T::create(exec, gko::dim<2>{0, 2})});
        }
        {
            SCOPED_TRACE("Uninitialized matrix (2x0)");
            guarded_fn(test_pair<Mtx>{T::create(ref, gko::dim<2>{2, 0}),
                                      T::create(exec, gko::dim<2>{2, 0})});
        }
        forall_matrix_data_scenarios([&](auto data) {
            test_pair<Mtx> pair{T::create(ref, data.size),
                                T::create(exec, data.size)};
            pair.dev->read(data);
            pair.ref->read(data);
            guarded_fn(std::move(pair));
        });
    }

    template <typename VecType = Vec, typename MtxType, typename TestFunction>
    void forall_vector_scenarios(const test_pair<MtxType>& mtx, TestFunction fn)
    {
        auto guarded_fn = [&](auto b, auto x) {
            try {
                fn(std::move(b), std::move(x));
            } catch (std::exception& e) {
                FAIL() << e.what();
            }
        };
        {
            SCOPED_TRACE("Multivector with 0 columns");
            guarded_fn(gen_in_vec<VecType>(mtx, 0, 0),
                       gen_out_vec<VecType>(mtx, 0, 0));
        }
        {
            SCOPED_TRACE("Single vector");
            guarded_fn(gen_in_vec<VecType>(mtx, 1, 1),
                       gen_out_vec<VecType>(mtx, 1, 1));
        }
        {
            SCOPED_TRACE("Single strided vector");
            guarded_fn(gen_in_vec<VecType>(mtx, 1, 2),
                       gen_out_vec<VecType>(mtx, 1, 3));
        }
        if (!gko::is_complex<value_type>()) {
            // check application of real matrix to complex vector
            // viewed as interleaved real/imag vector
            using complex_vec = gko::to_complex<VecType>;
            {
                SCOPED_TRACE("Single strided complex vector");
                guarded_fn(gen_in_vec<complex_vec>(mtx, 1, 2),
                           gen_out_vec<complex_vec>(mtx, 1, 3));
            }
            {
                SCOPED_TRACE("Strided complex multivector with 2 columns");
                guarded_fn(gen_in_vec<complex_vec>(mtx, 2, 3),
                           gen_out_vec<complex_vec>(mtx, 2, 4));
            }
        }
        {
            SCOPED_TRACE("Multivector with 2 columns");
            guarded_fn(gen_in_vec<VecType>(mtx, 2, 2),
                       gen_out_vec<VecType>(mtx, 2, 2));
        }
        {
            SCOPED_TRACE("Strided multivector with 2 columns");
            guarded_fn(gen_in_vec<VecType>(mtx, 2, 3),
                       gen_out_vec<VecType>(mtx, 2, 4));
        }
        {
            SCOPED_TRACE("Multivector with 40 columns");
            guarded_fn(gen_in_vec<VecType>(mtx, 40, 40),
                       gen_out_vec<VecType>(mtx, 40, 40));
        }
        {
            SCOPED_TRACE("Strided multivector with 40 columns");
            guarded_fn(gen_in_vec<VecType>(mtx, 40, 43),
                       gen_out_vec<VecType>(mtx, 40, 45));
        }
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::EXEC_TYPE> exec;

    std::default_random_engine rand_engine;
};

using MatrixTypes = ::testing::Types<
    DenseWithDefaultStride, DenseWithCustomStride, Coo, CsrWithDefaultStrategy,
    // The strategies have issues with zero rows
    /*
    #if defined(GKO_COMPILING_CUDA) || defined(GKO_COMPILING_HIP) || \
        defined(GKO_COMPILING_DPCPP)
        CsrWithClassicalStrategy, CsrWithMergePathStrategy,
        CsrWithSparselibStrategy, CsrWithLoadBalanceStrategy,
        CsrWithAutomaticalStrategy,
    #endif
    */
    Ell,
    // Fbcsr is slightly broken
    /*FbcsrBlocksize1, FbcsrBlocksize2,*/
    SellpDefaultParameters, Sellp32Factor2, HybridDefaultStrategy,
    HybridColumnLimitStrategy, HybridImbalanceLimitStrategy,
    HybridImbalanceBoundedLimitStrategy, HybridMinStorageStrategy,
    HybridAutomaticStrategy, SparsityCsr>;

TYPED_TEST_SUITE(Matrix, MatrixTypes, TypenameNameGenerator);


TYPED_TEST(Matrix, SpMVIsEquivalentToRef)
{
    this->forall_matrix_scenarios([&](auto mtx) {
        this->forall_vector_scenarios(mtx, [&](auto b, auto x) {
            mtx.ref->apply(b.ref.get(), x.ref.get());
            mtx.dev->apply(b.dev.get(), x.dev.get());

            GKO_ASSERT_MTX_NEAR(x.ref, x.dev, this->tol());
        });
    });
}


TYPED_TEST(Matrix, AdvancedSpMVIsEquivalentToRef)
{
    this->forall_matrix_scenarios([&](auto mtx) {
        this->forall_vector_scenarios(mtx, [&](auto b, auto x) {
            auto alpha = this->gen_scalar();
            auto beta = this->gen_scalar();

            mtx.ref->apply(alpha.ref.get(), b.ref.get(), alpha.ref.get(),
                           x.ref.get());
            mtx.dev->apply(alpha.dev.get(), b.dev.get(), alpha.dev.get(),
                           x.dev.get());

            GKO_ASSERT_MTX_NEAR(x.ref, x.dev, this->tol());
        });
    });
}


#if !(GINKGO_DPCPP_SINGLE_MODE)
TYPED_TEST(Matrix, MixedSpMVIsEquivalentToRef)
{
    using MixedVec = typename TestFixture::MixedVec;
    this->forall_matrix_scenarios([&](auto mtx) {
        this->template forall_vector_scenarios<MixedVec>(
            mtx, [&](auto b, auto x) {
                mtx.ref->apply(b.ref.get(), x.ref.get());
                mtx.dev->apply(b.dev.get(), x.dev.get());

                GKO_ASSERT_MTX_NEAR(x.ref, x.dev, this->mixed_tol());
            });
    });
}


TYPED_TEST(Matrix, MixedAdvancedSpMVIsEquivalentToRef)
{
    using MixedVec = typename TestFixture::MixedVec;
    this->forall_matrix_scenarios([&](auto mtx) {
        this->template forall_vector_scenarios<MixedVec>(
            mtx, [&](auto b, auto x) {
                auto alpha = this->template gen_scalar<MixedVec>();
                auto beta = this->template gen_scalar<MixedVec>();

                mtx.ref->apply(alpha.ref.get(), b.ref.get(), alpha.ref.get(),
                               x.ref.get());
                mtx.dev->apply(alpha.dev.get(), b.dev.get(), alpha.dev.get(),
                               x.dev.get());

                GKO_ASSERT_MTX_NEAR(x.ref, x.dev, this->mixed_tol());
            });
    });
}
#endif


TYPED_TEST(Matrix, ConvertToCsrIsEquivalentToRef)
{
    using Mtx = typename TestFixture::Mtx;
    using Csr =
        gko::matrix::Csr<typename Mtx::value_type, typename Mtx::index_type>;
    this->forall_matrix_scenarios([&](auto mtx) {
        auto ref_result = Csr::create(this->ref);
        auto dev_result = Csr::create(this->exec);

        mtx.ref->convert_to(ref_result.get());
        mtx.dev->convert_to(dev_result.get());

        GKO_ASSERT_MTX_NEAR(ref_result, dev_result, 0.0);
        GKO_ASSERT_MTX_EQ_SPARSITY(ref_result, dev_result);
    });
}


TYPED_TEST(Matrix, ConvertFromCsrIsEquivalentToRef)
{
    using TestConfig = typename TestFixture::Config;
    using Mtx = typename TestFixture::Mtx;
    using Csr =
        gko::matrix::Csr<typename Mtx::value_type, typename Mtx::index_type>;
    this->forall_matrix_data_scenarios([&](auto data) {
        auto ref_src = Csr::create(this->ref);
        auto dev_src = Csr::create(this->exec);
        ref_src->read(data);
        dev_src->read(data);
        auto ref_result = TestConfig::create(this->ref, data.size);
        auto dev_result = TestConfig::create(this->exec, data.size);

        ref_src->convert_to(ref_result.get());
        dev_src->convert_to(dev_result.get());

        GKO_ASSERT_MTX_NEAR(ref_result, dev_result, 0.0);
        GKO_ASSERT_MTX_EQ_SPARSITY(ref_result, dev_result);
    });
}


TYPED_TEST(Matrix, ConvertToDenseIsEquivalentToRef)
{
    using Mtx = typename TestFixture::Mtx;
    using Dense = gko::matrix::Dense<typename Mtx::value_type>;
    this->forall_matrix_scenarios([&](auto mtx) {
        auto ref_result = Dense::create(this->ref);
        auto dev_result = Dense::create(this->exec);

        mtx.ref->convert_to(ref_result.get());
        mtx.dev->convert_to(dev_result.get());

        GKO_ASSERT_MTX_NEAR(ref_result, dev_result, 0.0);
    });
}


TYPED_TEST(Matrix, ConvertFromDenseIsEquivalentToRef)
{
    using TestConfig = typename TestFixture::Config;
    using Mtx = typename TestFixture::Mtx;
    using Dense = gko::matrix::Dense<typename Mtx::value_type>;
    this->forall_matrix_data_scenarios([&](auto data) {
        auto ref_src = Dense::create(this->ref);
        auto dev_src = Dense::create(this->exec);
        ref_src->read(data);
        dev_src->read(data);
        auto ref_result = TestConfig::create(this->ref, data.size);
        auto dev_result = TestConfig::create(this->exec, data.size);

        ref_src->convert_to(ref_result.get());
        dev_src->convert_to(dev_result.get());

        GKO_ASSERT_MTX_NEAR(ref_result, dev_result, 0.0);
        GKO_ASSERT_MTX_EQ_SPARSITY(ref_result, dev_result);
    });
}


TYPED_TEST(Matrix, ReadWriteRoundtrip)
{
    using TestConfig = typename TestFixture::Config;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->forall_matrix_data_scenarios([&](auto data) {
        auto new_mtx = TestConfig::create(this->exec, data.size);
        gko::matrix_data<value_type, index_type> out_data;

        TestConfig::modify_data(data);
        new_mtx->read(data);
        new_mtx->write(out_data);

        if (!TestConfig::preserves_zeros()) {
            data.remove_zeros();
            out_data.remove_zeros();
        }
        ASSERT_EQ(data.size, out_data.size);
        ASSERT_EQ(data.nonzeros, out_data.nonzeros);
    });
}


TYPED_TEST(Matrix, DeviceReadCopyIsEquivalentToHostRef)
{
    using TestConfig = typename TestFixture::Config;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->forall_matrix_data_scenarios([&](auto data) {
        // put data on reference executor to test cross-executor execution
        const auto ref_device_data =
            gko::device_matrix_data<value_type, index_type>::create_from_host(
                this->ref, data);
        auto ref_result = TestConfig::create(this->ref, data.size);
        auto dev_result = TestConfig::create(this->exec, data.size);

        ref_result->read(data);
        dev_result->read(ref_device_data);

        GKO_ASSERT_MTX_NEAR(ref_result, dev_result, 0.0);
        GKO_ASSERT_MTX_EQ_SPARSITY(ref_result, dev_result);
    });
}


TYPED_TEST(Matrix, DeviceReadMoveIsEquivalentToHostRef)
{
    using TestConfig = typename TestFixture::Config;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->forall_matrix_data_scenarios([&](auto data) {
        // put data on reference executor to test cross-executor execution
        auto ref_device_data =
            gko::device_matrix_data<value_type, index_type>::create_from_host(
                this->ref, data);
        auto ref_result = TestConfig::create(this->ref, data.size);
        auto dev_result = TestConfig::create(this->exec, data.size);

        ref_result->read(data);
        dev_result->read(std::move(ref_device_data));

        ASSERT_EQ(ref_device_data.get_size(), gko::dim<2>{});
        ASSERT_EQ(ref_device_data.get_num_elems(), gko::dim<2>{});
        GKO_ASSERT_MTX_NEAR(ref_result, dev_result, 0.0);
        GKO_ASSERT_MTX_EQ_SPARSITY(ref_result, dev_result);
    });
}
