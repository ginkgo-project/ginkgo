/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <ginkgo/core/matrix/coo.hpp>


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


#include "core/matrix/coo_kernels.hpp"
#include "core/test/utils/unsort_matrix.hpp"
#include "cuda/test/utils.hpp"


namespace {


class Coo : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Coo<>;
    using Vec = gko::matrix::Dense<>;
    using ComplexVec = gko::matrix::Dense<std::complex<double>>;

    Coo() : rand_engine(42), ref_data(gko::ReferenceExecutor::create()) {}

    void SetUp()
    {
        auto ref = ref_data.exec;
#ifdef GKO_TEST_CUDA
        ASSERT_GT(gko::CudaExecutor::get_num_devices(), 0);
        device_data.emplace_back(gko::CudaExecutor::create(0, ref));
#endif
#ifdef GKO_TEST_HIP
        ASSERT_GT(gko::HipExecutor::get_num_devices(), 0);
        device_data.emplace_back(gko::HipExecutor::create(0, ref));
#endif
#ifdef GKO_TEST_OMP
        device_data.emplace_back(gko::OmpExecutor::create());
#endif
#ifdef GKO_TEST_DPCPP
        device_data.emplace_back(gko::DpcppExecutor::create(0, ref));
#endif
    }

    void TearDown()
    {
        for (const auto &dev : device_data) {
            if (dev.exec) {
                SCOPED_TRACE(dev.get_name());
                ASSERT_NO_THROW(dev.exec->synchronize());
            }
        }
    }

    template <typename MtxType = Vec>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols, std::uniform_int_distribution<>(1, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref_data.exec);
    }

    void set_up_apply_data(int num_vectors = 1)
    {
        auto ref = ref_data.exec;
        ref_data.mtx = Mtx::create(ref);
        ref_data.mtx->copy_from(gen_mtx(532, 231));
        ref_data.expected = gen_mtx(532, num_vectors);
        ref_data.y = gen_mtx(231, num_vectors);
        ref_data.complex_b = gen_mtx<ComplexVec>(231, num_vectors);
        ref_data.complex_x = gen_mtx<ComplexVec>(532, num_vectors);
        ref_data.alpha = gko::initialize<Vec>({2.0}, ref);
        ref_data.beta = gko::initialize<Vec>({-1.0}, ref);
    }

    void unsort_mtx()
    {
        gko::test::unsort_matrix(ref_data.mtx.get(), rand_engine);
    }

    struct data_struct {
        std::shared_ptr<gko::Executor> exec;
        std::unique_ptr<Mtx> mtx;
        std::unique_ptr<Vec> expected;
        std::unique_ptr<Vec> y;
        std::unique_ptr<Vec> alpha;
        std::unique_ptr<Vec> beta;
        std::unique_ptr<ComplexVec> complex_b;
        std::unique_ptr<ComplexVec> complex_x;

        data_struct(std::shared_ptr<gko::Executor> exec)
            : exec(exec),
              mtx(Mtx::create(exec)),
              expected(Vec::create(exec)),
              y(Vec::create(exec)),
              alpha(Vec::create(exec)),
              beta(Vec::create(exec)),
              complex_b(ComplexVec::create(exec)),
              complex_x(ComplexVec::create(exec))
        {}

        std::string get_name() const
        {
            return gko::name_demangling::get_dynamic_type(*exec);
        }

        void copy_from(const data_struct &other)
        {
            mtx->copy_from(other.mtx.get());
            expected->copy_from(other.expected.get());
            y->copy_from(other.y.get());
            alpha->copy_from(other.alpha.get());
            beta->copy_from(other.beta.get());
            complex_b->copy_from(other.complex_b.get());
            complex_x->copy_from(other.complex_x.get());
        }
    };

    template <typename RunFunc, typename TestFunc>
    void run_test(RunFunc run, TestFunc test)
    {
        for (auto &dev : device_data) {
            SCOPED_TRACE(dev.get_name());
            dev.copy_from(ref_data);
        }
        run(ref_data);
        for (auto &dev : device_data) {
            SCOPED_TRACE(dev.get_name());
            run(dev);
        }
        for (auto &dev : device_data) {
            SCOPED_TRACE(dev.get_name());
            test(dev, ref_data);
        }
    }

    template <typename RunFunc, typename TestFunc>
    void run_test_returning(RunFunc run, TestFunc test)
    {
        for (auto &dev : device_data) {
            SCOPED_TRACE(dev.get_name());
            dev.copy_from(ref_data);
        }
        auto ref_result = run(ref_data);
        std::vector<decltype(ref_result)> dev_results;
        for (auto &dev : device_data) {
            SCOPED_TRACE(dev.get_name());
            dev_results.push_back(run(dev));
        }
        int i = 0;
        for (auto &dev : device_data) {
            SCOPED_TRACE(dev.get_name());
            test(dev_results[i], ref_result);
            i++;
        }
    }

    std::ranlux48 rand_engine;
    data_struct ref_data;
    std::vector<data_struct> device_data;
};


TEST_F(Coo, SimpleApplyIsEquivalentToRef)
{
    set_up_apply_data();

    run_test(
        [](data_struct &data) {
            data.mtx->apply(data.y.get(), data.expected.get());
        },
        [](data_struct &dev, data_struct &ref) {
            GKO_ASSERT_MTX_NEAR(dev.expected, ref.expected, 1e-14);
        });
}


TEST_F(Coo, SimpleApplyIsEquivalentToRefUnsorted)
{
    set_up_apply_data();
    unsort_mtx();

    run_test(
        [](data_struct &data) {
            data.mtx->apply(data.y.get(), data.expected.get());
        },
        [](data_struct &dev, data_struct &ref) {
            GKO_ASSERT_MTX_NEAR(dev.expected, ref.expected, 1e-14);
        });
}


TEST_F(Coo, AdvancedApplyIsEquivalentToRef)
{
    set_up_apply_data();

    run_test(
        [](data_struct &data) {
            data.mtx->apply(data.alpha.get(), data.y.get(), data.beta.get(),
                            data.expected.get());
        },
        [](data_struct &dev, data_struct &ref) {
            GKO_ASSERT_MTX_NEAR(dev.expected, ref.expected, 1e-14);
        });
}


TEST_F(Coo, SimpleApplyAddIsEquivalentToRef)
{
    set_up_apply_data();

    run_test(
        [](data_struct &data) {
            data.mtx->apply2(data.y.get(), data.expected.get());
        },
        [](data_struct &dev, data_struct &ref) {
            GKO_ASSERT_MTX_NEAR(dev.expected, ref.expected, 1e-14);
        });
}


TEST_F(Coo, AdvancedApplyAddIsEquivalentToRef)
{
    set_up_apply_data();

    run_test(
        [](data_struct &data) {
            data.mtx->apply2(data.alpha.get(), data.y.get(),
                             data.expected.get());
        },
        [](data_struct &dev, data_struct &ref) {
            GKO_ASSERT_MTX_NEAR(dev.expected, ref.expected, 1e-14);
        });
}


TEST_F(Coo, SimpleApplyToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data(3);

    run_test(
        [](data_struct &data) {
            data.mtx->apply(data.y.get(), data.expected.get());
        },
        [](data_struct &dev, data_struct &ref) {
            GKO_ASSERT_MTX_NEAR(dev.expected, ref.expected, 1e-14);
        });
}


TEST_F(Coo, AdvancedApplyToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data(3);

    run_test(
        [](data_struct &data) {
            data.mtx->apply(data.alpha.get(), data.y.get(), data.beta.get(),
                            data.expected.get());
        },
        [](data_struct &dev, data_struct &ref) {
            GKO_ASSERT_MTX_NEAR(dev.expected, ref.expected, 1e-14);
        });
}


TEST_F(Coo, SimpleApplyAddToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data(3);

    run_test(
        [](data_struct &data) {
            data.mtx->apply2(data.y.get(), data.expected.get());
        },
        [](data_struct &dev, data_struct &ref) {
            GKO_ASSERT_MTX_NEAR(dev.expected, ref.expected, 1e-14);
        });
}


TEST_F(Coo, SimpleApplyAddToLargeDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data(33);

    run_test(
        [](data_struct &data) {
            data.mtx->apply2(data.y.get(), data.expected.get());
        },
        [](data_struct &dev, data_struct &ref) {
            GKO_ASSERT_MTX_NEAR(dev.expected, ref.expected, 1e-14);
        });
}


TEST_F(Coo, AdvancedApplyAddToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data(3);

    run_test(
        [](data_struct &data) {
            data.mtx->apply2(data.alpha.get(), data.y.get(),
                             data.expected.get());
        },
        [](data_struct &dev, data_struct &ref) {
            GKO_ASSERT_MTX_NEAR(dev.expected, ref.expected, 1e-14);
        });
}


TEST_F(Coo, AdvancedApplyAddToLargeDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data(33);

    run_test(
        [](data_struct &data) {
            data.mtx->apply2(data.alpha.get(), data.y.get(),
                             data.expected.get());
        },
        [](data_struct &dev, data_struct &ref) {
            GKO_ASSERT_MTX_NEAR(dev.expected, ref.expected, 1e-14);
        });
}


TEST_F(Coo, ApplyToComplexIsEquivalentToRef)
{
    set_up_apply_data(3);

    run_test(
        [](data_struct &data) {
            data.mtx->apply(data.complex_b.get(), data.complex_x.get());
        },
        [](data_struct &dev, data_struct &ref) {
            GKO_ASSERT_MTX_NEAR(dev.complex_x, ref.complex_x, 1e-14);
        });
}


TEST_F(Coo, AdvancedApplyToComplexIsEquivalentToRef)
{
    set_up_apply_data(3);

    run_test(
        [](data_struct &data) {
            data.mtx->apply(data.alpha.get(), data.complex_b.get(),
                            data.beta.get(), data.complex_x.get());
        },
        [](data_struct &dev, data_struct &ref) {
            GKO_ASSERT_MTX_NEAR(dev.complex_x, ref.complex_x, 1e-14);
        });
}


TEST_F(Coo, ApplyAddToComplexIsEquivalentToRef)
{
    set_up_apply_data(3);

    run_test(
        [](data_struct &data) {
            data.mtx->apply2(data.complex_b.get(), data.complex_x.get());
        },
        [](data_struct &dev, data_struct &ref) {
            GKO_ASSERT_MTX_NEAR(dev.complex_x, ref.complex_x, 1e-14);
        });
}


TEST_F(Coo, ConvertToDenseIsEquivalentToRef)
{
    set_up_apply_data();

    run_test_returning(
        [](data_struct &data) {
            auto dense_mtx = gko::matrix::Dense<>::create(data.exec);
            data.mtx->convert_to(dense_mtx.get());
            return dense_mtx;
        },
        [](auto &result, auto &ref_result) {
            GKO_ASSERT_MTX_NEAR(result.get(), ref_result.get(), 0);
        });
}


TEST_F(Coo, ConvertToCsrIsEquivalentToRef)
{
    set_up_apply_data();

    run_test_returning(
        [](data_struct &data) {
            auto csr_mtx = gko::matrix::Csr<>::create(data.exec);
            data.mtx->convert_to(csr_mtx.get());
            return csr_mtx;
        },
        [](auto &result, auto &ref_result) {
            GKO_ASSERT_MTX_NEAR(result.get(), ref_result.get(), 0);
        });
}


TEST_F(Coo, ExtractDiagonalIsEquivalentToRef)
{
    set_up_apply_data();

    run_test_returning(
        [](data_struct &data) { return data.mtx->extract_diagonal(); },
        [](auto &result, auto &ref_result) {
            GKO_ASSERT_MTX_NEAR(result.get(), ref_result.get(), 0);
        });
}


TEST_F(Coo, InplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_data();

    run_test([](data_struct &data) { data.mtx->compute_absolute_inplace(); },
             [](data_struct &dev, data_struct &ref) {
                 GKO_ASSERT_MTX_NEAR(dev.mtx, ref.mtx, 1e-14);
             });
}


TEST_F(Coo, OutplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_data();

    run_test_returning(
        [](data_struct &data) { return data.mtx->compute_absolute(); },
        [](auto &result, auto &ref_result) {
            GKO_ASSERT_MTX_NEAR(result, ref_result, 1e-14);
        });
}


}  // namespace
