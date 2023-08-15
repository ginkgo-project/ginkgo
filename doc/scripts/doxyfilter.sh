#!/bin/bash

# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

sed -e :a -e '/\\$/N;s/\\\n */ /;ta' $1 | sed '/#/d;/^\w*$/d' | sort
