# Pretty-printers for Ginkgo
# Based on the pretty-printers for libstdc++.

# Copyright (C) 2008-2020 Free Software Foundation, Inc.

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
    ### Python 3 stuff
    Iterator = object
    # Python 3 folds these into the normal functions.
    imap = map
    izip = zip
    # Also, int subsumes long
    long = int
else:
    ### Python 2 stuff
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

    # In Python 2, we still need these from itertools
    from itertools import imap, izip

_versioned_namespace = '__8::'

def is_specialization_of(x, template_name):
    "Test if a type is a given template instantiation."
    global _versioned_namespace
    if type(x) is gdb.Type:
        x = x.tag
    if _versioned_namespace:
        return re.match('^std::(%s)?%s<.*>$' % (_versioned_namespace, template_name), x) is not None
    return re.match('^std::%s<.*>$' % template_name, x) is not None


def get_unique_ptr_data_ptr(val):
    impl_type = val.type.fields()[0].type.tag
    # Check for new implementations first:
    if is_specialization_of(impl_type, '__uniq_ptr_data') \
        or is_specialization_of(impl_type, '__uniq_ptr_impl'):
        tuple_member = val['_M_t']['_M_t']
    elif is_specialization_of(impl_type, 'tuple'):
        tuple_member = val['_M_t']
    else:
        raise ValueError("Unsupported implementation for unique_ptr: %s" % impl_type)
    tuple_impl_type = tuple_member.type.fields()[0].type # _Tuple_impl
    tuple_head_type = tuple_impl_type.fields()[1].type   # _Head_base
    head_field = tuple_head_type.fields()[0]
    if head_field.name == '_M_head_impl':
        return tuple_member['_M_head_impl']
    elif head_field.is_base_class:
        return tuple_member.cast(head_field.type)
    else:
        raise ValueError("Unsupported implementation for tuple in unique_ptr: %s" % impl_type)


class GkoArrayPrinter:
    "Print a gko::Array"

    class _iterator(Iterator):
        def __init__ (self, start, size):
            self.item = start
            self.size = size
            self.count = 0

        def __iter__(self):
            return self

        def __next__(self):
            count = self.count
            self.count = self.count + 1
            if self.count >= self.size:
                raise StopIteration
            elt = self.item.dereference()
            self.item = self.item + 1
            return ('[%d]' % count, elt)

    def __init__(self, val):
        self.val = val
        self.execname = str(self.val['exec_']['_M_ptr'].dereference().dynamic_type)
        self.pointer = get_unique_ptr_data_ptr(self.val['data_']);
        self.is_cpu = re.match('gko::(Reference|Omp)Executor', str(self.execname)) is not None

    def children(self):
        if self.is_cpu:
            return self._iterator(self.pointer, self.val['num_elems_'])
        return []

    def to_string(self):     
        return ('%s of length %d on %s (%s)' % (str(self.val.type), int(self.val['num_elems_']), self.execname, self.pointer))

    def display_hint(self):
        return 'array'

def lookup_type(val):
    if not str(val.type).startswith('gko::'):
        return None
    suffix = str(val.type)[5:]
    if suffix.startswith('Array'):
        return GkoArrayPrinter(val)
    return None

gdb.pretty_printers.append(lookup_type)
