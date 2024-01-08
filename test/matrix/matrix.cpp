// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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


#if GINKGO_COMMON_SINGLE_MODE
using matrix_value_type = float;
#else
using matrix_value_type = double;
#endif  // GINKGO_COMMON_SINGLE_MODE


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

    static void check_property(gko::ptr_param<const matrix_type>) {}

    static bool supports_strides() { return true; }

    static void assert_empty_state(gko::ptr_param<const matrix_type> mtx)
    {
        ASSERT_FALSE(mtx->get_size());
    }
};

struct DenseWithDefaultStride
    : SimpleMatrixTest<gko::matrix::Dense<matrix_value_type>> {
    static bool preserves_zeros() { return false; }

    static void assert_empty_state(gko::ptr_param<const matrix_type> mtx)
    {
        ASSERT_FALSE(mtx->get_size());
        ASSERT_EQ(mtx->get_stride(), 0);
        ASSERT_EQ(mtx->get_num_stored_elements(), 0);
        ASSERT_EQ(mtx->get_const_values(), nullptr);
    }
};

struct DenseWithCustomStride : DenseWithDefaultStride {
    static std::unique_ptr<matrix_type> create(
        std::shared_ptr<gko::Executor> exec, gko::dim<2> size)
    {
        return matrix_type::create(exec, size, size[0] + 10);
    }

    static void check_property(gko::ptr_param<const matrix_type> mtx)
    {
        ASSERT_EQ(mtx->get_stride(), mtx->get_size()[0] + 10);
    }

    static void assert_empty_state(gko::ptr_param<const matrix_type> mtx)
    {
        ASSERT_FALSE(mtx->get_size());
        ASSERT_EQ(mtx->get_stride(), 0);
        ASSERT_EQ(mtx->get_num_stored_elements(), 0);
        ASSERT_EQ(mtx->get_const_values(), nullptr);
    }
};

struct Coo : SimpleMatrixTest<gko::matrix::Coo<matrix_value_type, int>> {
    static void assert_empty_state(gko::ptr_param<const matrix_type> mtx)
    {
        ASSERT_FALSE(mtx->get_size());
        ASSERT_EQ(mtx->get_num_stored_elements(), 0);
        ASSERT_EQ(mtx->get_const_row_idxs(), nullptr);
        ASSERT_EQ(mtx->get_const_col_idxs(), nullptr);
        ASSERT_EQ(mtx->get_const_values(), nullptr);
    }
};

struct CsrBase : SimpleMatrixTest<gko::matrix::Csr<matrix_value_type, int>> {
    static void assert_empty_state(gko::ptr_param<const matrix_type> mtx)
    {
        ASSERT_FALSE(mtx->get_size());
        ASSERT_EQ(mtx->get_num_stored_elements(), 0);
        ASSERT_NE(mtx->get_const_row_ptrs(), nullptr);
        ASSERT_EQ(
            mtx->get_executor()->copy_val_to_host(mtx->get_const_row_ptrs()),
            0);
        ASSERT_EQ(mtx->get_const_col_idxs(), nullptr);
        ASSERT_EQ(mtx->get_const_values(), nullptr);
    }
};

struct CsrWithDefaultStrategy : CsrBase {
    static void assert_empty_state(gko::ptr_param<const matrix_type> mtx)
    {
        CsrBase::assert_empty_state(mtx);
        auto first_strategy = mtx->create_default()->get_strategy();
        auto second_strategy = mtx->get_strategy();
        GKO_ASSERT_DYNAMIC_TYPE_EQ(first_strategy, second_strategy);
    }
};


#if defined(GKO_COMPILING_CUDA) || defined(GKO_COMPILING_HIP) || \
    defined(GKO_COMPILING_DPCPP)


struct CsrWithClassicalStrategy : CsrBase {
    static std::unique_ptr<matrix_type> create(
        std::shared_ptr<gko::Executor> exec, gko::dim<2> size)
    {
        return matrix_type::create(exec, size, 0,
                                   std::make_shared<matrix_type::classical>());
    }

    static void check_property(gko::ptr_param<const matrix_type> mtx)
    {
        ASSERT_TRUE(dynamic_cast<const matrix_type::classical*>(
            mtx->get_strategy().get()));
    }

    static void assert_empty_state(gko::ptr_param<const matrix_type> mtx)
    {
        CsrBase::assert_empty_state(mtx);
        ASSERT_TRUE(dynamic_cast<const matrix_type::classical*>(
            mtx->get_strategy().get()));
    }
};

struct CsrWithMergePathStrategy : CsrBase {
    static std::unique_ptr<matrix_type> create(
        std::shared_ptr<gko::Executor> exec, gko::dim<2> size)
    {
        return matrix_type::create(exec, size, 0,
                                   std::make_shared<matrix_type::merge_path>());
    }

    static void check_property(gko::ptr_param<const matrix_type> mtx)
    {
        ASSERT_TRUE(dynamic_cast<const matrix_type::merge_path*>(
            mtx->get_strategy().get()));
    }

    static void assert_empty_state(gko::ptr_param<const matrix_type> mtx)
    {
        CsrBase::assert_empty_state(mtx);
        ASSERT_TRUE(dynamic_cast<const matrix_type::merge_path*>(
            mtx->get_strategy().get()));
    }
};

struct CsrWithSparselibStrategy : CsrBase {
    static std::unique_ptr<matrix_type> create(
        std::shared_ptr<gko::Executor> exec, gko::dim<2> size)
    {
        return matrix_type::create(exec, size, 0,
                                   std::make_shared<matrix_type::sparselib>());
    }

    static void check_property(gko::ptr_param<const matrix_type> mtx)
    {
        ASSERT_TRUE(dynamic_cast<const matrix_type::sparselib*>(
            mtx->get_strategy().get()));
    }

    static void assert_empty_state(gko::ptr_param<const matrix_type> mtx)
    {
        CsrBase::assert_empty_state(mtx);
        ASSERT_TRUE(dynamic_cast<const matrix_type::sparselib*>(
            mtx->get_strategy().get()));
    }
};

struct CsrWithLoadBalanceStrategy : CsrBase {
    static std::unique_ptr<matrix_type> create(
        std::shared_ptr<gko::Executor> exec, gko::dim<2> size)
    {
        return matrix_type::create(exec, size, 0,
                                   std::make_shared<matrix_type::load_balance>(
                                       gko::EXEC_TYPE::create(0, exec)));
    }

    static void check_property(gko::ptr_param<const matrix_type> mtx)
    {
        ASSERT_TRUE(dynamic_cast<const matrix_type::load_balance*>(
            mtx->get_strategy().get()));
    }

    static void assert_empty_state(gko::ptr_param<const matrix_type> mtx)
    {
        CsrBase::assert_empty_state(mtx);
        ASSERT_TRUE(dynamic_cast<const matrix_type::load_balance*>(
            mtx->get_strategy().get()));
    }
};

struct CsrWithAutomaticalStrategy : CsrBase {
    static std::unique_ptr<matrix_type> create(
        std::shared_ptr<gko::Executor> exec, gko::dim<2> size)
    {
        return matrix_type::create(exec, size, 0,
                                   std::make_shared<matrix_type::automatical>(
                                       gko::EXEC_TYPE::create(0, exec)));
    }

    static void check_property(gko::ptr_param<const matrix_type> mtx)
    {
        ASSERT_TRUE(dynamic_cast<const matrix_type::automatical*>(
            mtx->get_strategy().get()));
    }

    static void assert_empty_state(gko::ptr_param<const matrix_type> mtx)
    {
        CsrBase::assert_empty_state(mtx);
        ASSERT_TRUE(dynamic_cast<const matrix_type::automatical*>(
            mtx->get_strategy().get()));
    }
};


#endif


struct Ell : SimpleMatrixTest<gko::matrix::Ell<matrix_value_type, int>> {
    static void assert_empty_state(gko::ptr_param<const matrix_type> mtx)
    {
        ASSERT_FALSE(mtx->get_size());
        ASSERT_EQ(mtx->get_num_stored_elements_per_row(), 0);
        ASSERT_EQ(mtx->get_num_stored_elements(), 0);
        ASSERT_EQ(mtx->get_stride(), 0);
        ASSERT_EQ(mtx->get_const_col_idxs(), nullptr);
        ASSERT_EQ(mtx->get_const_values(), nullptr);
    }
};


template <int block_size>
struct Fbcsr : SimpleMatrixTest<gko::matrix::Fbcsr<matrix_value_type, int>> {
    static bool preserves_zeros() { return false; }

    static std::unique_ptr<matrix_type> create(
        std::shared_ptr<gko::Executor> exec, gko::dim<2> size)
    {
        size[0] = gko::ceildiv(size[0], block_size) * block_size;
        size[1] = gko::ceildiv(size[1], block_size) * block_size;
        return matrix_type::create(exec, size, 0, block_size);
    }

    static void check_property(gko::ptr_param<const matrix_type> mtx)
    {
        ASSERT_EQ(mtx->get_block_size(), block_size);
    }

    static void assert_empty_state(gko::ptr_param<const matrix_type> mtx)
    {
        ASSERT_FALSE(mtx->get_size());
        ASSERT_EQ(mtx->get_block_size(), block_size);
    }

    static void modify_data(gko::matrix_data<matrix_value_type, int>& data)
    {
        data.size[0] = gko::ceildiv(data.size[0], block_size) * block_size;
        data.size[1] = gko::ceildiv(data.size[1], block_size) * block_size;
    }

#ifdef GKO_COMPILING_HIP
    // Fbcsr support in rocSPARSE is buggy w.r.t. strides
    static bool supports_strides() { return false; }
#endif
};


struct SellpBase
    : SimpleMatrixTest<gko::matrix::Sellp<matrix_value_type, int>> {
    static void assert_empty_state(gko::ptr_param<const matrix_type> mtx)
    {
        ASSERT_FALSE(mtx->get_size());
        ASSERT_EQ(mtx->get_num_stored_elements(), 0);
        ASSERT_EQ(mtx->get_total_cols(), 0);
        ASSERT_NE(mtx->get_const_slice_sets(), nullptr);
        ASSERT_EQ(
            mtx->get_executor()->copy_val_to_host(mtx->get_const_slice_sets()),
            0);
        ASSERT_EQ(mtx->get_const_slice_lengths(), nullptr);
        ASSERT_EQ(mtx->get_const_col_idxs(), nullptr);
        ASSERT_EQ(mtx->get_const_values(), nullptr);
    }
};


struct SellpDefaultParameters : SellpBase {
    static void check_property(gko::ptr_param<const matrix_type> mtx)
    {
        ASSERT_EQ(mtx->get_stride_factor(), 1);
        ASSERT_EQ(mtx->get_slice_size(), 64);
    }

    static void assert_empty_state(gko::ptr_param<const matrix_type> mtx)
    {
        SellpBase::assert_empty_state(mtx);
        ASSERT_EQ(mtx->get_stride_factor(), 1);
        ASSERT_EQ(mtx->get_slice_size(), 64);
    }
};

struct Sellp32Factor2 : SellpBase {
    static std::unique_ptr<matrix_type> create(
        std::shared_ptr<gko::Executor> exec, gko::dim<2> size)
    {
        return matrix_type::create(exec, size, 32, 2, 0);
    }

    static void check_property(gko::ptr_param<const matrix_type> mtx)
    {
        ASSERT_EQ(mtx->get_stride_factor(), 2);
        ASSERT_EQ(mtx->get_slice_size(), 32);
    }

    static void assert_empty_state(gko::ptr_param<const matrix_type> mtx)
    {
        SellpBase::assert_empty_state(mtx);
        ASSERT_EQ(mtx->get_stride_factor(), 2);
        ASSERT_EQ(mtx->get_slice_size(), 32);
    }
};


struct HybridBase
    : SimpleMatrixTest<gko::matrix::Hybrid<matrix_value_type, int>> {
    static void assert_empty_state(gko::ptr_param<const matrix_type> mtx)
    {
        ASSERT_FALSE(mtx->get_size());
        ASSERT_FALSE(mtx->get_coo()->get_size());
        ASSERT_EQ(mtx->get_coo_num_stored_elements(), 0);
        ASSERT_EQ(mtx->get_const_coo_row_idxs(), nullptr);
        ASSERT_EQ(mtx->get_const_coo_col_idxs(), nullptr);
        ASSERT_EQ(mtx->get_const_coo_values(), nullptr);
        ASSERT_FALSE(mtx->get_ell()->get_size());
        ASSERT_EQ(mtx->get_ell_num_stored_elements_per_row(), 0);
        ASSERT_EQ(mtx->get_ell_num_stored_elements(), 0);
        ASSERT_EQ(mtx->get_ell_stride(), 0);
        ASSERT_EQ(mtx->get_const_ell_col_idxs(), nullptr);
        ASSERT_EQ(mtx->get_const_ell_values(), nullptr);
    }
};


struct HybridDefaultStrategy : HybridBase {
    static void check_property(gko::ptr_param<const matrix_type> mtx)
    {
        auto strategy = dynamic_cast<const matrix_type::automatic*>(
            mtx->get_strategy().get());
        ASSERT_TRUE(strategy);
    }

    static void assert_empty_state(gko::ptr_param<const matrix_type> mtx)
    {
        HybridBase::assert_empty_state(mtx);
        check_property(mtx);
    }
};

struct HybridColumnLimitStrategy : HybridBase {
    static std::unique_ptr<matrix_type> create(
        std::shared_ptr<gko::Executor> exec, gko::dim<2> size)
    {
        return matrix_type::create(
            exec, size, 0, std::make_shared<matrix_type::column_limit>(10));
    }

    static void check_property(gko::ptr_param<const matrix_type> mtx)
    {
        auto strategy = dynamic_cast<const matrix_type::column_limit*>(
            mtx->get_strategy().get());
        ASSERT_TRUE(strategy);
        ASSERT_EQ(strategy->get_num_columns(), 10);
    }

    static void assert_empty_state(gko::ptr_param<const matrix_type> mtx)
    {
        HybridBase::assert_empty_state(mtx);
        check_property(mtx);
    }
};

struct HybridImbalanceLimitStrategy : HybridBase {
    static std::unique_ptr<matrix_type> create(
        std::shared_ptr<gko::Executor> exec, gko::dim<2> size)
    {
        return matrix_type::create(
            exec, size, 0, std::make_shared<matrix_type::imbalance_limit>(0.5));
    }

    static void check_property(gko::ptr_param<const matrix_type> mtx)
    {
        auto strategy = dynamic_cast<const matrix_type::imbalance_limit*>(
            mtx->get_strategy().get());
        ASSERT_TRUE(strategy);
        ASSERT_EQ(strategy->get_percentage(), 0.5);
    }

    static void assert_empty_state(gko::ptr_param<const matrix_type> mtx)
    {
        HybridBase::assert_empty_state(mtx);
        check_property(mtx);
    }
};

struct HybridImbalanceBoundedLimitStrategy : HybridBase {
    static std::unique_ptr<matrix_type> create(
        std::shared_ptr<gko::Executor> exec, gko::dim<2> size)
    {
        return matrix_type::create(
            exec, size, 0,
            std::make_shared<matrix_type::imbalance_bounded_limit>(0.5, 0.01));
    }

    static void check_property(gko::ptr_param<const matrix_type> mtx)
    {
        auto strategy =
            dynamic_cast<const matrix_type::imbalance_bounded_limit*>(
                mtx->get_strategy().get());
        ASSERT_TRUE(strategy);
        ASSERT_EQ(strategy->get_percentage(), 0.5);
        ASSERT_EQ(strategy->get_ratio(), 0.01);
    }

    static void assert_empty_state(gko::ptr_param<const matrix_type> mtx)
    {
        HybridBase::assert_empty_state(mtx);
        check_property(mtx);
    }
};

struct HybridMinStorageStrategy : HybridBase {
    static std::unique_ptr<matrix_type> create(
        std::shared_ptr<gko::Executor> exec, gko::dim<2> size)
    {
        return matrix_type::create(
            exec, size, 0,
            std::make_shared<matrix_type::minimal_storage_limit>());
    }

    static void check_property(gko::ptr_param<const matrix_type> mtx)
    {
        auto strategy = dynamic_cast<const matrix_type::minimal_storage_limit*>(
            mtx->get_strategy().get());
        ASSERT_TRUE(strategy);
    }

    static void assert_empty_state(gko::ptr_param<const matrix_type> mtx)
    {
        HybridBase::assert_empty_state(mtx);
        check_property(mtx);
    }
};

struct HybridAutomaticStrategy : HybridBase {
    static std::unique_ptr<matrix_type> create(
        std::shared_ptr<gko::Executor> exec, gko::dim<2> size)
    {
        return matrix_type::create(exec, size, 0,
                                   std::make_shared<matrix_type::automatic>());
    }

    static void check_property(gko::ptr_param<const matrix_type> mtx)
    {
        auto strategy = dynamic_cast<const matrix_type::automatic*>(
            mtx->get_strategy().get());
        ASSERT_TRUE(strategy);
    }

    static void assert_empty_state(gko::ptr_param<const matrix_type> mtx)
    {
        HybridBase::assert_empty_state(mtx);
        check_property(mtx);
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

    static void assert_empty_state(gko::ptr_param<const matrix_type> mtx)
    {
        ASSERT_FALSE(mtx->get_size());
        ASSERT_EQ(mtx->get_num_nonzeros(), 0);
        ASSERT_NE(mtx->get_const_row_ptrs(), nullptr);
        ASSERT_EQ(
            mtx->get_executor()->copy_val_to_host(mtx->get_const_row_ptrs()),
            0);
        ASSERT_EQ(mtx->get_const_col_idxs(), nullptr);
        ASSERT_NE(mtx->get_const_value(), nullptr);
        ASSERT_EQ(mtx->get_executor()->copy_val_to_host(mtx->get_const_value()),
                  gko::one<matrix_value_type>());
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
class Matrix : public CommonTestFixture {
protected:
    using Config = T;
    using Mtx = typename T::matrix_type;
    using index_type = typename Mtx::index_type;
    using value_type = typename Mtx::value_type;
    using mixed_value_type = gko::next_precision<value_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using MixedVec = gko::matrix::Dense<mixed_value_type>;

    Matrix() : rand_engine(15) {}

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
    test_pair<VecType> gen_in_vec(const test_pair<Mtx>& mtx, int nrhs)
    {
        auto size = gko::dim<2>{mtx.ref->get_size()[1],
                                static_cast<gko::size_type>(nrhs)};
        auto result = VecType::create(ref, size);
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
    test_pair<VecType> gen_out_vec(const test_pair<Mtx>& mtx, int nrhs)
    {
        auto size = gko::dim<2>{mtx.ref->get_size()[0],
                                static_cast<gko::size_type>(nrhs)};
        auto result = VecType::create(ref, size);
        result->read(gen_dense_data<typename VecType::value_type,
                                    typename Mtx::index_type>(size));
        return {std::move(result), exec};
    }

    double tol() { return r<value_type>::value; }

    double mixed_tol() { return r_mixed<value_type, mixed_value_type>(); }

    template <typename TestFunction>
    void forall_matrix_data_scenarios(TestFunction fn)
    {
        auto guarded_fn = [&](auto data) {
            try {
                Config::modify_data(data);
                fn(std::move(data));
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
            SCOPED_TRACE("Small Sparse Matrix with variable row nnz (10x10)");
            guarded_fn(gen_mtx_data(10, 10, 1, 5));
        }
        {
            SCOPED_TRACE("Small Dense Matrix (10x10)");
            guarded_fn(gen_mtx_data(10, 10, 10, 10));
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
                "Sparse Matrix with variable row nnz and some numerical zeros "
                "(200x100)");
            auto data = gen_mtx_data(200, 100, 10, 50);
            for (int i = 0; i < data.nonzeros.size() / 4; i++) {
                data.nonzeros[i * 4].value = gko::zero<value_type>();
            }
            guarded_fn(data);
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

    template <typename VecType = Vec, typename OutVecType = VecType,
              typename MtxType, typename TestFunction>
    void run_strided(const test_pair<MtxType>& mtx, int rhs, int in_stride,
                     int out_stride, TestFunction fn)
    {
        // create slightly bigger vectors
        auto in_padded = gen_in_vec<VecType>(mtx, in_stride);
        auto out_padded = gen_out_vec<OutVecType>(mtx, out_stride);
        const auto in_rows = gko::span(0, mtx.ref->get_size()[1]);
        const auto out_rows = gko::span(0, mtx.ref->get_size()[0]);
        const auto cols = gko::span(0, rhs);
        const auto out_pad_cols = gko::span(rhs, out_stride);
        // create views of the padding and in/out vectors
        auto out_padding = test_pair<OutVecType>{
            out_padded.ref->create_submatrix(out_rows, out_pad_cols),
            out_padded.dev->create_submatrix(out_rows, out_pad_cols)};
        auto orig_padding = out_padding.ref->clone();
        auto in =
            test_pair<VecType>{in_padded.ref->create_submatrix(in_rows, cols),
                               in_padded.dev->create_submatrix(in_rows, cols)};
        auto out = test_pair<OutVecType>{
            out_padded.ref->create_submatrix(out_rows, cols),
            out_padded.dev->create_submatrix(out_rows, cols)};
        fn(std::move(in), std::move(out));
        // check that padding was unmodified
        GKO_ASSERT_MTX_NEAR(out_padding.ref, orig_padding, 0.0);
        GKO_ASSERT_MTX_NEAR(out_padding.dev, orig_padding, 0.0);
    }

    template <typename VecType = Vec, typename OutVecType = VecType,
              typename MtxType, typename TestFunction>
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
            guarded_fn(gen_in_vec<VecType>(mtx, 0),
                       gen_out_vec<OutVecType>(mtx, 0));
        }
        {
            SCOPED_TRACE("Single vector");
            guarded_fn(gen_in_vec<VecType>(mtx, 1),
                       gen_out_vec<OutVecType>(mtx, 1));
        }
        if (Config::supports_strides()) {
            SCOPED_TRACE("Single strided vector");
            run_strided<VecType, OutVecType>(mtx, 1, 2, 3, guarded_fn);
        }
        if (!gko::is_complex<value_type>()) {
            // check application of real matrix to complex vector
            // viewed as interleaved real/imag vector
            using complex_vec = gko::to_complex<VecType>;
            using complex_out_vec = gko::to_complex<OutVecType>;
            if (Config::supports_strides()) {
                SCOPED_TRACE("Single strided complex vector");
                run_strided<complex_vec, complex_out_vec>(mtx, 1, 2, 3,
                                                          guarded_fn);
            }
            if (Config::supports_strides()) {
                SCOPED_TRACE("Strided complex multivector with 2 columns");
                run_strided<complex_vec, complex_out_vec>(mtx, 2, 3, 4,
                                                          guarded_fn);
            }
        }
        {
            SCOPED_TRACE("Multivector with 2 columns");
            guarded_fn(gen_in_vec<VecType>(mtx, 2),
                       gen_out_vec<OutVecType>(mtx, 2));
        }
        if (Config::supports_strides()) {
            SCOPED_TRACE("Strided multivector with 2 columns");
            run_strided<VecType, OutVecType>(mtx, 2, 3, 4, guarded_fn);
        }
        {
            SCOPED_TRACE("Multivector with 40 columns");
            guarded_fn(gen_in_vec<VecType>(mtx, 40),
                       gen_out_vec<OutVecType>(mtx, 40));
        }
        if (Config::supports_strides()) {
            SCOPED_TRACE("Strided multivector with 40 columns");
            run_strided<VecType, OutVecType>(mtx, 40, 43, 45, guarded_fn);
        }
    }

    std::default_random_engine rand_engine;
};

using MatrixTypes = ::testing::Types<
    DenseWithDefaultStride, DenseWithCustomStride, Coo, CsrWithDefaultStrategy,
#if defined(GKO_COMPILING_CUDA) || defined(GKO_COMPILING_HIP) || \
    defined(GKO_COMPILING_DPCPP)
    CsrWithClassicalStrategy, CsrWithMergePathStrategy,
    CsrWithSparselibStrategy, CsrWithLoadBalanceStrategy,
    CsrWithAutomaticalStrategy,
#endif
    Ell,
#ifdef GKO_COMPILING_OMP
    // CUDA doesn't allow blocksize 1
    Fbcsr<1>,
#endif
#if defined(GKO_COMPILING_OMP) || defined(GKO_COMPILING_CUDA) || \
    defined(GKO_COMPILING_HIP)
    Fbcsr<2>, Fbcsr<3>,
#endif
    SellpDefaultParameters, Sellp32Factor2, HybridDefaultStrategy,
    HybridColumnLimitStrategy, HybridImbalanceLimitStrategy,
    HybridImbalanceBoundedLimitStrategy, HybridMinStorageStrategy,
    HybridAutomaticStrategy, SparsityCsr>;

TYPED_TEST_SUITE(Matrix, MatrixTypes, TypenameNameGenerator);


TYPED_TEST(Matrix, SpMVIsEquivalentToRef)
{
    this->forall_matrix_scenarios([&](auto mtx) {
        this->forall_vector_scenarios(mtx, [&](auto b, auto x) {
            mtx.ref->apply(b.ref, x.ref);
            mtx.dev->apply(b.dev, x.dev);

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

            mtx.ref->apply(alpha.ref, b.ref, alpha.ref, x.ref);
            mtx.dev->apply(alpha.dev, b.dev, alpha.dev, x.dev);

            GKO_ASSERT_MTX_NEAR(x.ref, x.dev, this->tol());
        });
    });
}


#if !(GINKGO_COMMON_SINGLE_MODE)
TYPED_TEST(Matrix, MixedSpMVIsEquivalentToRef)
{
    using MixedVec = typename TestFixture::MixedVec;
    this->forall_matrix_scenarios([&](auto mtx) {
        this->template forall_vector_scenarios<MixedVec>(
            mtx, [&](auto b, auto x) {
                mtx.ref->apply(b.ref, x.ref);
                mtx.dev->apply(b.dev, x.dev);

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

                mtx.ref->apply(alpha.ref, b.ref, alpha.ref, x.ref);
                mtx.dev->apply(alpha.dev, b.dev, alpha.dev, x.dev);

                GKO_ASSERT_MTX_NEAR(x.ref, x.dev, this->mixed_tol());
            });
    });
}


TYPED_TEST(Matrix, MixedInputSpMVIsEquivalentToRef)
{
    using Vec = typename TestFixture::Vec;
    using MixedVec = typename TestFixture::MixedVec;
    this->forall_matrix_scenarios([&](auto mtx) {
        this->template forall_vector_scenarios<MixedVec, Vec>(
            mtx, [&](auto b, auto x) {
                mtx.ref->apply(b.ref, x.ref);
                mtx.dev->apply(b.dev, x.dev);

                GKO_ASSERT_MTX_NEAR(x.ref, x.dev, this->tol());
            });
    });
}


TYPED_TEST(Matrix, MixedInputAdvancedSpMVIsEquivalentToRef)
{
    using Vec = typename TestFixture::Vec;
    using MixedVec = typename TestFixture::MixedVec;
    this->forall_matrix_scenarios([&](auto mtx) {
        this->template forall_vector_scenarios<MixedVec, Vec>(
            mtx, [&](auto b, auto x) {
                auto alpha = this->template gen_scalar<MixedVec>();
                auto beta = this->template gen_scalar<Vec>();

                mtx.ref->apply(alpha.ref, b.ref, alpha.ref, x.ref);
                mtx.dev->apply(alpha.dev, b.dev, alpha.dev, x.dev);

                GKO_ASSERT_MTX_NEAR(x.ref, x.dev, this->tol());
            });
    });
}


TYPED_TEST(Matrix, MixedOutputSpMVIsEquivalentToRef)
{
    using Vec = typename TestFixture::Vec;
    using MixedVec = typename TestFixture::MixedVec;
    this->forall_matrix_scenarios([&](auto mtx) {
        this->template forall_vector_scenarios<Vec, MixedVec>(
            mtx, [&](auto b, auto x) {
                mtx.ref->apply(b.ref, x.ref);
                mtx.dev->apply(b.dev, x.dev);

                GKO_ASSERT_MTX_NEAR(x.ref, x.dev, this->mixed_tol());
            });
    });
}


TYPED_TEST(Matrix, MixedOutputAdvancedSpMVIsEquivalentToRef)
{
    using Vec = typename TestFixture::Vec;
    using MixedVec = typename TestFixture::MixedVec;
    this->forall_matrix_scenarios([&](auto mtx) {
        this->template forall_vector_scenarios<Vec, MixedVec>(
            mtx, [&](auto b, auto x) {
                auto alpha = this->template gen_scalar<Vec>();
                auto beta = this->template gen_scalar<MixedVec>();

                mtx.ref->apply(alpha.ref, b.ref, alpha.ref, x.ref);
                mtx.dev->apply(alpha.dev, b.dev, alpha.dev, x.dev);

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

        mtx.ref->convert_to(ref_result);
        mtx.dev->convert_to(dev_result);

        GKO_ASSERT_MTX_NEAR(ref_result, dev_result, 0.0);
        GKO_ASSERT_MTX_EQ_SPARSITY(ref_result, dev_result);
    });
}


TYPED_TEST(Matrix, MoveToCsrIsEquivalentToRef)
{
    using Mtx = typename TestFixture::Mtx;
    using Csr =
        gko::matrix::Csr<typename Mtx::value_type, typename Mtx::index_type>;
    this->forall_matrix_scenarios([&](auto mtx) {
        auto ref_result = Csr::create(this->ref);
        auto dev_result = Csr::create(this->exec);

        mtx.ref->move_to(ref_result);
        mtx.dev->move_to(dev_result);

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

        ref_src->convert_to(ref_result);
        dev_src->convert_to(dev_result);

        GKO_ASSERT_MTX_NEAR(ref_result, dev_result, 0.0);
        GKO_ASSERT_MTX_EQ_SPARSITY(ref_result, dev_result);
    });
}


TYPED_TEST(Matrix, MoveFromCsrIsEquivalentToRef)
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

        ref_src->move_to(ref_result);
        dev_src->move_to(dev_result);

        GKO_ASSERT_MTX_NEAR(ref_result, dev_result, 0.0);
        GKO_ASSERT_MTX_EQ_SPARSITY(ref_result, dev_result);
    });
}


TYPED_TEST(Matrix, ConvertToDenseIsEquivalentToRef)
{
    using Mtx = typename TestFixture::Mtx;
    using Dense = gko::matrix::Dense<typename Mtx::value_type>;
    this->forall_matrix_scenarios([&](auto mtx) {
        const auto size = mtx.ref->get_size();
        const auto stride = size[1] + 5;
        const auto padded_size = gko::dim<2>{size[0], stride};
        auto ref_padded = Dense::create(this->ref, padded_size);
        auto dev_padded = Dense::create(this->exec, padded_size);
        ref_padded->fill(12345);
        dev_padded->fill(12345);
        const auto rows = gko::span{0, size[0]};
        const auto cols = gko::span{0, size[1]};
        const auto pad_cols = gko::span{size[1], stride};
        auto ref_result = ref_padded->create_submatrix(rows, cols);
        auto dev_result = dev_padded->create_submatrix(rows, cols);
        auto ref_padding = ref_padded->create_submatrix(rows, pad_cols);
        auto dev_padding = dev_padded->create_submatrix(rows, pad_cols);
        auto orig_padding = ref_padding->clone();

        mtx.ref->convert_to(ref_result);
        mtx.dev->convert_to(dev_result);

        GKO_ASSERT_MTX_NEAR(ref_result, dev_result, 0.0);
        ASSERT_EQ(ref_result->get_stride(), stride);
        ASSERT_EQ(dev_result->get_stride(), stride);
        GKO_ASSERT_MTX_NEAR(ref_padding, orig_padding, 0.0);
        GKO_ASSERT_MTX_NEAR(dev_padding, orig_padding, 0.0);
    });
}


TYPED_TEST(Matrix, MoveToDenseIsEquivalentToRef)
{
    using Mtx = typename TestFixture::Mtx;
    using Dense = gko::matrix::Dense<typename Mtx::value_type>;
    this->forall_matrix_scenarios([&](auto mtx) {
        auto ref_result = Dense::create(this->ref);
        auto dev_result = Dense::create(this->exec);

        mtx.ref->move_to(ref_result);
        mtx.dev->move_to(dev_result);

        GKO_ASSERT_MTX_NEAR(ref_result, dev_result, 0.0);
    });
}


TYPED_TEST(Matrix, ConvertFromDenseIsEquivalentToRef)
{
    using TestConfig = typename TestFixture::Config;
    using Mtx = typename TestFixture::Mtx;
    using Dense = gko::matrix::Dense<typename Mtx::value_type>;
    this->forall_matrix_data_scenarios([&](auto data) {
        const auto stride = data.size[0] + 2;
        auto ref_src = Dense::create(this->ref, data.size, stride);
        auto dev_src = Dense::create(this->exec, data.size, stride);
        ref_src->read(data);
        dev_src->read(data);
        ASSERT_EQ(ref_src->get_stride(), stride);
        ASSERT_EQ(dev_src->get_stride(), stride);
        auto ref_result = TestConfig::create(this->ref, data.size);
        auto dev_result = TestConfig::create(this->exec, data.size);

        ref_src->convert_to(ref_result);
        dev_src->convert_to(dev_result);

        GKO_ASSERT_MTX_NEAR(ref_result, dev_result, 0.0);
        GKO_ASSERT_MTX_EQ_SPARSITY(ref_result, dev_result);
    });
}


TYPED_TEST(Matrix, MoveFromDenseIsEquivalentToRef)
{
    using TestConfig = typename TestFixture::Config;
    using Mtx = typename TestFixture::Mtx;
    using Dense = gko::matrix::Dense<typename Mtx::value_type>;
    this->forall_matrix_data_scenarios([&](auto data) {
        const auto stride = data.size[0] + 2;
        auto ref_src = Dense::create(this->ref, data.size, stride);
        auto dev_src = Dense::create(this->exec, data.size, stride);
        ref_src->read(data);
        dev_src->read(data);
        ASSERT_EQ(ref_src->get_stride(), stride);
        ASSERT_EQ(dev_src->get_stride(), stride);
        auto ref_result = TestConfig::create(this->ref, data.size);
        auto dev_result = TestConfig::create(this->exec, data.size);

        ref_src->move_to(ref_result);
        dev_src->move_to(dev_result);

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
        ASSERT_EQ(ref_device_data.get_num_stored_elements(), 0);
        GKO_ASSERT_MTX_NEAR(ref_result, dev_result, 0.0);
        GKO_ASSERT_MTX_EQ_SPARSITY(ref_result, dev_result);
    });
}


TYPED_TEST(Matrix, CopyAssignIsCorrect)
{
    using TestConfig = typename TestFixture::Config;
    using Mtx = typename TestFixture::Mtx;
    this->forall_matrix_data_scenarios([&](auto data) {
        auto mtx = TestConfig::create(this->exec, data.size);
        auto mtx2 = Mtx::create(this->exec);
        mtx->read(data);

        auto& result = (*mtx2 = *mtx);

        ASSERT_EQ(&result, mtx2.get());
        GKO_ASSERT_MTX_NEAR(mtx, mtx2, 0.0);
        GKO_ASSERT_MTX_EQ_SPARSITY(mtx, mtx2);
    });
}


TYPED_TEST(Matrix, MoveAssignIsCorrect)
{
    using TestConfig = typename TestFixture::Config;
    using Mtx = typename TestFixture::Mtx;
    this->forall_matrix_data_scenarios([&](auto data) {
        auto mtx = TestConfig::create(this->exec, data.size);
        auto mtx2 = Mtx::create(this->exec);
        mtx->read(data);
        auto orig_mtx = mtx->clone();

        auto& result = (*mtx2 = std::move(*mtx));

        ASSERT_EQ(&result, mtx2.get());
        GKO_ASSERT_MTX_NEAR(mtx2, orig_mtx, 0.0);
        GKO_ASSERT_MTX_EQ_SPARSITY(mtx2, orig_mtx);
        TestConfig::assert_empty_state(mtx);
    });
}


TYPED_TEST(Matrix, CopyAssignToDifferentExecutorIsCorrect)
{
    using TestConfig = typename TestFixture::Config;
    using Mtx = typename TestFixture::Mtx;
    this->forall_matrix_data_scenarios([&](auto data) {
        auto mtx = TestConfig::create(this->exec, data.size);
        auto mtx2 = Mtx::create(this->ref);
        mtx->read(data);

        auto& result = (*mtx2 = *mtx);

        ASSERT_EQ(&result, mtx2.get());
        GKO_ASSERT_MTX_NEAR(mtx, mtx2, 0.0);
        GKO_ASSERT_MTX_EQ_SPARSITY(mtx, mtx2);
    });
}


TYPED_TEST(Matrix, MoveAssignToDifferentExecutorIsCorrect)
{
    using TestConfig = typename TestFixture::Config;
    using Mtx = typename TestFixture::Mtx;
    this->forall_matrix_data_scenarios([&](auto data) {
        auto mtx = TestConfig::create(this->exec, data.size);
        auto mtx2 = Mtx::create(this->ref);
        mtx->read(data);
        auto orig_mtx = mtx->clone();

        auto& result = (*mtx2 = std::move(*mtx));

        ASSERT_EQ(&result, mtx2.get());
        GKO_ASSERT_MTX_NEAR(mtx2, orig_mtx, 0.0);
        GKO_ASSERT_MTX_EQ_SPARSITY(mtx2, orig_mtx);
        TestConfig::assert_empty_state(mtx);
    });
}
