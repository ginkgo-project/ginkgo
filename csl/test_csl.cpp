// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <csl/Engine.hpp>
#include <csl/IPUModel.hpp>

using namespace csl;
using namespace csl::program;

int main()
{
    IPUModel ipuModel;
    Device device = ipuModel.createDevice();
    Target target = device.getTarget();

    return 0;
}
