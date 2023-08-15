#!/bin/bash

# SPDX-FileCopyrightText: 2013 - 2019 by the deal.II authors
#
# SPDX-License-Identifier: LGPL-2.1-or-later

cd $1 && git log --format="format:%h" -n1 -- $2
