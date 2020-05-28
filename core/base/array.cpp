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

#include <ginkgo/core/base/array.hpp>


#include <ginkgo/core/base/math.hpp>


#include "core/components/precision_conversion_kernels.hpp"


namespace gko {
namespace conversion {


GKO_REGISTER_OPERATION(convert, components::convert_precision);


}  // namespace conversion


namespace detail {


template <typename SourceType, typename TargetType>
void convert_data(const std::shared_ptr<const Executor> &exec, size_type size,
                  const SourceType *src, TargetType *dst)
{
    exec->run(conversion::make_convert(size, src, dst));
}


#define GKO_DECLARE_ARRAY_CONVERSION(From, To)                           \
    void convert_data<From, To>(const std::shared_ptr<const Executor> &, \
                                size_type, const From *, To *)

GKO_INSTANTIATE_FOR_EACH_VALUE_CONVERSION(GKO_DECLARE_ARRAY_CONVERSION);


}  // namespace detail
}  // namespace gko
