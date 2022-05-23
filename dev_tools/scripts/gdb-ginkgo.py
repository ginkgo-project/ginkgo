# Pretty-printers for Ginkgo
#
# Usage inside gdb:
# > source path/to/ginkgo/dev_tools/scripts/gdb-ginkgo.py
#   load the pretty-printer
# > print object->array_
#   print the contents of the given array
# > set print elements 1000
#   limit the output to 1000 elements
#
# Based on the pretty-printers for libstdc++.

# Copyright (C) 2008-2020 Free Software Foundation, Inc.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import gdb
import itertools
import sys
import re

if sys.version_info[0] > 2:
    # Python 3 stuff
    Iterator = object
else:
    # Python 2 stuff
    class Iterator:
        """Compatibility mixin for iterators

        Instead of writing next() methods for iterators, write
        __next__() methods and use this mixin to make them work in
        Python 2 as well as Python 3.

        Idea stolen from the "six" documentation:
        <http://pythonhosted.org/six/#six.Iterator>
        """

        def next(self):
            return self.__next__()

_versioned_namespace = '__8::'


def is_specialization_of(x, template_name):
    "Test if a type is a given template instantiation."
    global _versioned_namespace
    if type(x) is gdb.Type:
        x = x.tag
    if _versioned_namespace:
        expr = '^std::({})?{}<.*>$'.format(_versioned_namespace, template_name)
    else:
        expr = '^std::{}<.*>$'.format(template_name)
    return re.match(expr, x) is not None


def get_unique_ptr_data_ptr(val):
    impl_type = val.type.fields()[0].type.tag
    # Check for new implementations first:
    if is_specialization_of(impl_type, '__uniq_ptr_data') \
            or is_specialization_of(impl_type, '__uniq_ptr_impl'):
        tuple_member = val['_M_t']['_M_t']
    elif is_specialization_of(impl_type, 'tuple'):
        tuple_member = val['_M_t']
    else:
        raise ValueError(
            "Unsupported unique_ptr impl: {}".format(impl_type))
    tuple_impl_type = tuple_member.type.fields()[0].type  # _Tuple_impl
    tuple_head_type = tuple_impl_type.fields()[1].type    # _Head_base
    head_field = tuple_head_type.fields()[0]
    if head_field.name == '_M_head_impl':
        return tuple_member['_M_head_impl']
    elif head_field.is_base_class:
        return tuple_member.cast(head_field.type)
    else:
        raise ValueError(
            "Unsupported tuple impl in unique_ptr: {}".format(impl_type))


class GkoArrayPrinter:
    "Print a gko::array"

    class _iterator(Iterator):
        def __init__(self, exec, start, size):
            self.exec = exec
            self.item = start
            self.size = size
            self.count = 0
            if exec in ["gko::CudaExecutor", "gko::HipExecutor"]:
                self.sizeof = self.item.dereference().type.sizeof
                self.buffer_start = 0
                # At most 1 MB or size, at least 1
                self.buffer_size = min(size, max(1, 2 ** 20 // self.sizeof))
                self.buffer = gdb.parse_and_eval(
                    "(void*)malloc({})".format(self.buffer_size * self.sizeof))
                self.buffer.fetch_lazy()
                self.buffer_count = self.buffer_size
                self.update_buffer()
            else:
                self.buffer = None

        def update_buffer(self):
            if self.buffer and self.buffer_count >= self.buffer_size:
                self.buffer_item = gdb.parse_and_eval(
                    hex(self.buffer)).cast(self.item.type)
                self.buffer_count = 0
                self.buffer_start = self.count
                cuda = "(cudaError)cudaMemcpy({},{},{},cudaMemcpyDeviceToHost)"
                hip = "(hipError_t)hipMemcpy({},{},{},hipMemcpyDeviceToHost)"
                if self.exec == "gko::CudaExecutor":
                    memcpy_expr = cuda
                elif self.exec == "gko::HipExecutor":
                    memcpy_expr = hip
                else:
                    raise StopIteration
                device_addr = hex(self.item.dereference().address)
                buffer_addr = hex(self.buffer)
                size = min(self.buffer_size, self.size -
                           self.buffer_start) * self.sizeof
                status = gdb.parse_and_eval(
                    memcpy_expr.format(buffer_addr, device_addr, size))
                if status != 0:
                    raise gdb.MemoryError(
                        "memcpy from device failed: {}".format(status))

        def __del__(self):
            if self.buffer:
                gdb.parse_and_eval("(void)free({})".format(
                    hex(self.buffer))).fetch_lazy()

        def __iter__(self):
            return self

        def __next__(self):
            if self.count >= self.size:
                raise StopIteration
            if self.buffer:
                self.update_buffer()
                elt = self.buffer_item.dereference()
                self.buffer_item += 1
                self.buffer_count += 1
            else:
                elt = self.item.dereference()
            count = self.count
            self.item += 1
            self.count += 1
            return ('[{}]'.format(count), elt)

    def __init__(self, val):
        self.val = val
        self.execname = str(
            self.val['exec_']['_M_ptr']
            .dereference()
            .dynamic_type
            .unqualified())
        self.pointer = get_unique_ptr_data_ptr(self.val['data_'])

    def children(self):
        return self._iterator(self.execname,
                              self.pointer,
                              self.val['num_elems_'])

    def to_string(self):
        return ('{} of length {} on {} ({})'
                .format(str(self.val.type),
                        int(self.val['num_elems_']),
                        self.execname,
                        self.pointer))

    def display_hint(self):
        return 'array'


def lookup_type(val):
    if not str(val.type.unqualified()).startswith('gko::'):
        return None
    suffix = str(val.type.unqualified())[5:]
    if suffix.startswith('array'):
        return GkoArrayPrinter(val)
    return None


gdb.pretty_printers.append(lookup_type)
