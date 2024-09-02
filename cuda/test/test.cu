// SPDX-FileCopyrightText: 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <iostream>
#include <limits>

#include <thrust/device_vector.h>

__global__ void isnan_direct(double* d) { *d = isnan(*d); }

__host__ __device__ bool is_nan(double d)
{
    using std::isnan;
    return isnan(d);
}

__global__ void isnan_indirect(double* d) { *d = is_nan(*d); }

int main()
{
    thrust::device_vector<double> vec(2);
    vec[0] = std::numeric_limits<double>::quiet_NaN();
    vec[1] = std::numeric_limits<double>::quiet_NaN();
    isnan_direct<<<1, 1>>>(vec.data().get());
    isnan_indirect<<<1, 1>>>(vec.data().get() + 1);
    std::cout << vec[0] << vec[1] << '\n';
    if (vec[0] != vec[1]) {
        return 1;
    }
}
