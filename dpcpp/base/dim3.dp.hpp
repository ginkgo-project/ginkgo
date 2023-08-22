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

#ifndef GKO_DPCPP_BASE_DIM3_DP_HPP_
#define GKO_DPCPP_BASE_DIM3_DP_HPP_


#include <CL/sycl.hpp>


namespace gko {
namespace kernels {
namespace sycl {


/**
 * dim3 is a cuda-like dim3 for sycl-range, which provides the same ordering as
 * cuda and gets the sycl-range in reverse ordering.
 */
struct dim3 {
    unsigned int x;
    unsigned int y;
    unsigned int z;

    /**
     * Creates a dim3 with x, y, z
     *
     * @param xval  x dim val
     * @param yval  y dim val and default is 1
     * @param zval  z dim val and default is 1
     */
    dim3(unsigned int xval, unsigned int yval = 1, unsigned int zval = 1)
        : x(xval), y(yval), z(zval)
    {}

    /**
     * get_range returns the range for sycl with correct ordering (reverse of
     * cuda)
     *
     * @return ::sycl::range<3>
     */
    ::sycl::range<3> get_range() { return ::sycl::range<3>(z, y, x); }
};


/**
 * sycl_nd_range will generate the proper sycl::nd_range<3> from grid, block
 *
 * @param grid  the dim3 for grid
 * @param block  the dim3 for block
 *
 * @return sycl::nd_range<3>
 */
inline ::sycl::nd_range<3> sycl_nd_range(dim3 grid, dim3 block)
{
    auto local_range = block.get_range();
    auto global_range = grid.get_range() * local_range;
    return ::sycl::nd_range<3>(global_range, local_range);
}


}  // namespace sycl
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_BASE_DIM3_DP_HPP_
