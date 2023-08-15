#!/bin/bash

# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

cd $1 && git log --format="format:%h" -n1 -- $2
