// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

// Configure-time probe: reports the resolved Highway static target and the
// number of double lanes, so a silently under-vectorized build is visible.
// Deliberately header-only: libhwy may not be built when this runs.
#include <cstdio>

#include <hwy/highway.h>

#define GKO_PROBE_STR2(x) #x
#define GKO_PROBE_STR(x) GKO_PROBE_STR2(x)

int main()
{
    namespace hn = hwy::HWY_NAMESPACE;
    std::printf("%s %zu\n", GKO_PROBE_STR(HWY_NAMESPACE),
                static_cast<std::size_t>(hn::Lanes(hn::ScalableTag<double>())));
    return 0;
}
