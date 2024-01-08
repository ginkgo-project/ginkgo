// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_BASE_DIM3_DP_HPP_
#define GKO_DPCPP_BASE_DIM3_DP_HPP_


#include <CL/sycl.hpp>


namespace gko {
namespace kernels {
namespace dpcpp {


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
     * @return sycl::range<3>
     */
    sycl::range<3> get_range() { return sycl::range<3>(z, y, x); }
};


/**
 * sycl_nd_range will generate the proper sycl::nd_range<3> from grid, block
 *
 * @param grid  the dim3 for grid
 * @param block  the dim3 for block
 *
 * @return sycl::nd_range<3>
 */
inline sycl::nd_range<3> sycl_nd_range(dim3 grid, dim3 block)
{
    auto local_range = block.get_range();
    auto global_range = grid.get_range() * local_range;
    return sycl::nd_range<3>(global_range, local_range);
}


}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_BASE_DIM3_DP_HPP_
