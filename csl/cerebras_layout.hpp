// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <string>

struct CerebrasLayout {
    std::string name;
    int offset1;
    int offset2;
    int size1;
    int size2;
    int elements_per_pe;
    bool streaming;
    bool nonblocking;

    CerebrasLayout(std::string name, int offset1, int offset2, int size1,
                   int size2, int elements_per_pe, bool streaming,
                   bool nonblocking)
        : offset1(offset1),
          offset2(offset2),
          size1(size1),
          size2(size2),
          elements_per_pe(elements_per_pe),
          streaming(streaming),
          nonblocking(nonblocking)
    {}
};
