#!/bin/bash

# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

cat $1 | perl -0777 -pe 's|<<<.*?>>>||smg'
