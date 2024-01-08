// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef TRICK_DIM3_T_HPP_
#define TRICK_DIM3_T_HPP_

struct dim3_t {
    unsigned int x;
    unsigned int y;
    unsigned int z;

    dim3_t(unsigned int xval, unsigned int yval = 1, unsigned int zval = 1) : x(xval), y(yval), z(zval) {}

    operator dim3() { return dim3{x, y, z}; }
};

#endif  // TRICK_DIM3_T_HPP_
