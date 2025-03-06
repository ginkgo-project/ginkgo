// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>

using namespace poplar;
using namespace poplar::program;

int main()
{
    IPUModel ipuModel;
    Device device = ipuModel.createDevice();
    Target target = device.getTarget();

    return 0;
}
