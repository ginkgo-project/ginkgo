#!/bin/env bash

# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

INPUT_IMAGE=$1
SIZES=256,128,64,48,32,16

convert "${INPUT_IMAGE}" -define icon:auto-resize="$SIZES" favicon.ico
