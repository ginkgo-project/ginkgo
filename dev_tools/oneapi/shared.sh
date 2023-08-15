#!/bin/bash

# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

# Checks if $1 is self contained code, that is it does not have an open and
# unclosed code portion (<>()[]), e.g. `my_struct->my_func(xxx,` should fail.
check_closed() {
    local str="$1"
    # remove -> to avoid the confusion
    str="${str//->}"
    # Replace everything except begin or end characters, resp. (<[ and )>]
    str_start="${str//[^(<\[]}"
    str_end="${str//[^>)\]]}"
    # Check that there are as many begin as end characters
    if [[ "${#str_start}" -eq "${#str_end}" ]]; then
        echo "true"
    else
        echo "false"
    fi
}

GINKGO_LICENSE_BEACON="******************************<GINKGO LICENSE>******************************"
HOST_SUFFIX="_AUTOHOSTFUNC"
MAP_FILE="map_list"
