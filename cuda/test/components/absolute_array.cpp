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

#include "core/components/absolute_array.hpp"


#include <memory>
#include <random>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>


#include "core/test/utils/assertions.hpp"


namespace {


class AbsoluteArray : public ::testing::Test {
protected:
    using value_type = double;
    using complex_type = std::complex<double>;
    AbsoluteArray()
        : ref(gko::ReferenceExecutor::create()),
          exec(gko::CudaExecutor::create(0, ref)),
          total_size(6344),
          vals(ref, total_size),
          dvals(exec, total_size),
          complex_vals(ref, total_size),
          dcomplex_vals(exec, total_size)
    {
        std::fill_n(vals.get_data(), total_size, -1234.0);
        dvals = vals;
        std::fill_n(complex_vals.get_data(), total_size, complex_type{3, 4});
        dcomplex_vals = complex_vals;
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::CudaExecutor> exec;
    gko::size_type total_size;
    gko::Array<value_type> vals;
    gko::Array<value_type> dvals;
    gko::Array<complex_type> complex_vals;
    gko::Array<complex_type> dcomplex_vals;
};


TEST_F(AbsoluteArray, InplaceEqualsReference)
{
    gko::kernels::cuda::components::inplace_absolute_array(
        exec, dvals.get_data(), total_size);
    gko::kernels::reference::components::inplace_absolute_array(
        ref, vals.get_data(), total_size);

    GKO_ASSERT_ARRAY_EQ(vals, dvals);
}


TEST_F(AbsoluteArray, InplaceComplexEqualsReference)
{
    gko::kernels::cuda::components::inplace_absolute_array(
        exec, dcomplex_vals.get_data(), total_size);
    gko::kernels::reference::components::inplace_absolute_array(
        ref, complex_vals.get_data(), total_size);

    GKO_ASSERT_ARRAY_EQ(complex_vals, dcomplex_vals);
}


TEST_F(AbsoluteArray, OutplaceEqualsReference)
{
    gko::Array<value_type> abs_vals(ref, total_size);
    gko::Array<value_type> dabs_vals(exec, total_size);

    gko::kernels::cuda::components::outplace_absolute_array(
        exec, dvals.get_const_data(), total_size, dabs_vals.get_data());
    gko::kernels::reference::components::outplace_absolute_array(
        ref, vals.get_const_data(), total_size, abs_vals.get_data());

    GKO_ASSERT_ARRAY_EQ(abs_vals, dabs_vals);
}


TEST_F(AbsoluteArray, OutplaceComplexEqualsReference)
{
    gko::Array<value_type> abs_vals(ref, total_size);
    gko::Array<value_type> dabs_vals(exec, total_size);

    gko::kernels::cuda::components::outplace_absolute_array(
        exec, dcomplex_vals.get_const_data(), total_size, dabs_vals.get_data());
    gko::kernels::reference::components::outplace_absolute_array(
        ref, complex_vals.get_const_data(), total_size, abs_vals.get_data());

    GKO_ASSERT_ARRAY_EQ(abs_vals, dabs_vals);
}


}  // namespace
