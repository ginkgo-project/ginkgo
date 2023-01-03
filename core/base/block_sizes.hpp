/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#ifndef GKO_CORE_BASE_BLOCK_SIZES_HPP_
#define GKO_CORE_BASE_BLOCK_SIZES_HPP_


#include <ginkgo/config.hpp>
#include <ginkgo/core/synthesizer/containers.hpp>


namespace gko {
namespace fixedblock {


/**
 * @def GKO_FIXED_BLOCK_CUSTOM_SIZES
 * Optionally-defined comma-separated list of fixed block sizes to compile.
 */
#ifdef GKO_FIXED_BLOCK_CUSTOM_SIZES
/**
 * A compile-time list of block sizes for which dedicated fixed-block matrix
 * and corresponding preconditioner kernels should be compiled.
 */
using compiled_kernels = syn::value_list<int, GKO_FIXED_BLOCK_CUSTOM_SIZES>;
#else
using compiled_kernels = syn::value_list<int, 2, 3, 4, 7>;
#endif


}  // namespace fixedblock
}  // namespace gko


#endif  // GKO_CORE_BASE_BLOCK_SIZES_HPP_
