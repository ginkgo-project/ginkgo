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


#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>
#include <ginkgo/core/matrix/sellp.hpp>


#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


namespace {


template <typename ObjectType>
struct test_pair {
    std::unique_ptr<ObjectType> ref;
    std::unique_ptr<ObjectType> dev;

    test_pair(std::unique_ptr<ObjectType> ref_obj,
              std::shared_ptr<const gko::Executor> exec)
        : ref{std::move(ref_obj)}, dev{gko::clone(exec, ref)}
    {}
};


template <typename T>
class Matrix : public ::testing::Test {
protected:
    using Mtx = T;
    using index_type = typename Mtx::index_type;
    using value_type = typename Mtx::value_type;
    using mixed_value_type = gko::next_precision<value_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using ComplexVec = gko::to_complex<Vec>;
    using MixedVec = gko::matrix::Dense<mixed_value_type>;
    using MixedComplexVec = gko::to_complex<MixedVec>;

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

    template <typename MtxType = Mtx, typename DistType>
    test_pair<MtxType> gen_mtx(int num_rows, int num_cols, DistType dist)
    {
        return {gko::test::generate_random_matrix<MtxType>(
                    num_rows, num_cols, dist,
                    std::normal_distribution<>(0.0, 1.0), rand_engine, ref),
                exec};
    }

    template <typename MtxType = Mtx>
    test_pair<MtxType> gen_mtx(int num_rows, int num_cols, int min_cols,
                               int max_cols)
    {
        return gen_mtx<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(min_cols, max_cols));
    }

    template <typename ValueType, typename IndexType>
    gko::matrix_data<ValueType, IndexType> gen_dense_data(gko::dim<2> size)
    {
        return {size, std::normal_distribution<>(0.0, 1.0), rand_engine};
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
                        std::normal_distribution<>(0.0, 1.0), rand_engine)},
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

    template <typename MtxType = Mtx, typename TestFunction>
    void forall_matrix_scenarios(TestFunction fn)
    {
        auto guarded_fn = [&](auto mtx) {
            try {
                fn(std::move(mtx));
            } catch (std::exception& e) {
                FAIL() << e.what();
            }
        };
        {
            SCOPED_TRACE("Uninitialized matrix (0x0)");
            guarded_fn(test_pair<MtxType>{MtxType::create(ref), exec});
        }
        {
            SCOPED_TRACE("Uninitialized matrix (0x1)");
            guarded_fn(test_pair<MtxType>{
                MtxType::create(ref, gko::dim<2>{0, 1}), exec});
        }
        {
            SCOPED_TRACE("Uninitialized matrix (1x0)");
            guarded_fn(test_pair<MtxType>{
                MtxType::create(ref, gko::dim<2>{1, 0}), exec});
        }
        {
            SCOPED_TRACE("Zero matrix (0x0)");
            guarded_fn(gen_mtx<MtxType>(0, 0, 0, 0));
        }
        {
            SCOPED_TRACE("Zero matrix (0x1)");
            guarded_fn(gen_mtx<MtxType>(0, 0, 0, 0));
        }
        {
            SCOPED_TRACE("Zero matrix (1x0)");
            guarded_fn(gen_mtx<MtxType>(0, 0, 0, 0));
        }
        {
            SCOPED_TRACE("Zero matrix (200x100)");
            guarded_fn(gen_mtx<MtxType>(200, 100, 0, 0));
        }
        {
            SCOPED_TRACE("Sparse Matrix with some zeros rows (200x100)");
            guarded_fn(gen_mtx<MtxType>(200, 100, 0, 50));
        }
        {
            SCOPED_TRACE("Sparse Matrix with fixed row nnz (200x100)");
            guarded_fn(gen_mtx<MtxType>(200, 100, 50, 50));
        }
        {
            SCOPED_TRACE("Sparse Matrix with variable row nnz (200x100)");
            guarded_fn(gen_mtx<MtxType>(200, 100, 10, 50));
        }
        {
            SCOPED_TRACE(
                "Sparse Matrix with heavily imbalanced row nnz (200x100)");
            guarded_fn(gen_mtx<MtxType>(
                200, 100, std::binomial_distribution<>{100, 0.05}));
        }
        {
            SCOPED_TRACE("Dense matrix (200x100)");
            guarded_fn(gen_mtx<MtxType>(200, 100, 100, 100));
        }
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

    std::ranlux48 rand_engine;
};

using MatrixTypes = ::testing::Types<
    gko::matrix::Dense<double>, gko::matrix::Coo<double, int>,
    gko::matrix::Csr<double, int>,
    gko::matrix::Ell<double, int>  //,   gko::matrix::Sellp<double, int>,
    // gko::matrix::Fbcsr<double, int>
    >;

// TODO remove
struct MatrixTypeNameGenerator {
    template <typename T>
    static std::string GetName(int i)
    {
        return gko::name_demangling::get_type_name(typeid(T));
    }
};

TYPED_TEST_SUITE(Matrix, MatrixTypes, MatrixTypeNameGenerator);


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


TYPED_TEST(Matrix, ConvertToCsrIsEquivalentToRef)
{
    using Csr = gko::matrix::Csr<typename TypeParam::value_type,
                                 typename TypeParam::index_type>;
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
    using Csr = gko::matrix::Csr<typename TypeParam::value_type,
                                 typename TypeParam::index_type>;
    using Mtx = TypeParam;
    this->template forall_matrix_scenarios<Csr>([&](auto mtx) {
        auto ref_result = Mtx::create(this->ref);
        auto dev_result = Mtx::create(this->exec);

        mtx.ref->convert_to(ref_result.get());
        mtx.dev->convert_to(dev_result.get());

        GKO_ASSERT_MTX_NEAR(ref_result, dev_result, 0.0);
        GKO_ASSERT_MTX_EQ_SPARSITY(ref_result, dev_result);
    });
}


TYPED_TEST(Matrix, ConvertToDenseIsEquivalentToRef)
{
    using Dense = gko::matrix::Dense<typename TypeParam::value_type>;
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
    using Dense = gko::matrix::Dense<typename TypeParam::value_type>;
    using Mtx = TypeParam;
    this->template forall_matrix_scenarios<Dense>([&](auto mtx) {
        auto ref_result = Mtx::create(this->ref);
        auto dev_result = Mtx::create(this->exec);

        mtx.ref->convert_to(ref_result.get());
        mtx.dev->convert_to(dev_result.get());

        GKO_ASSERT_MTX_NEAR(ref_result, dev_result, 0.0);
        GKO_ASSERT_MTX_EQ_SPARSITY(ref_result, dev_result);
    });
}


TYPED_TEST(Matrix, WriteReadRoundtrip)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename Mtx::value_type;
    using index_type = typename Mtx::index_type;
    this->forall_matrix_scenarios([&](auto mtx) {
        auto new_mtx = Mtx::create(this->exec);
        gko::matrix_data<value_type, index_type> data;

        mtx.dev->write(data);
        new_mtx->read(data);

        GKO_ASSERT_MTX_NEAR(mtx.dev, new_mtx, 0.0);
        GKO_ASSERT_MTX_EQ_SPARSITY(mtx.dev, new_mtx);
    });
}


}  //  namespace
