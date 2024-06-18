.. NeoFOAM documentation master file, created by
   sphinx-quickstart on Sat Dec 16 15:22:11 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
Welcome to NeoFOAM!
===================

The NeoFOAM project has set itself the goal of bringing modern software development methods to the core libraries of OpenFOAM.
By reimplementing the `libfiniteVolume` and `libOpenFOAM` we want to deliver a code that:

* is compliant with modern C++20;
* is extensively unit-tested;
* is platform portable and GPU ready;
* is highly extensible via Plugins

We aim for a high level of interoperability with OpenFOAM, however, if reasonable, NeoFOAM might deviate from the OpenFOAM API. NeoFOAM is a community-driven project and we welcome contributions from everyone.

Table of Contents
^^^^^^^^^^^^^^^^^
.. toctree::
   :maxdepth: 2

   self
   installation
   contributing
   basics/index
   fvcc/index
   api/index

Compatibility with OpenFOAM
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We are aiming for a high level of compatibility with OpenFOAM. However, we don't expect binary or ABI compatibility. This means you NeoFOAM won't produce a `libfiniteVolume.so` and `libOpenFOAM.so` which could serve as a plugin replacement for existing `libfiniteVolume.so` and `libOpenFOAM.so`. Instead, we aim for source compatibility, i.e. the possibility to compile application OpenFOAM code like pimpleFoam  and others against the NeoFOAM libraries.

Building OpenFOAM Applications with NeoFOAM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, the support for building applications is very limited. The simplest way to build applications
with the NeoFOAM core is by adopting the CMake build procedure showcased in our `applications <https://github.com/exasim-project/NeoFOAM/tree/main/applications>`_
folder. For now, we only have a `minimal <https://github.com/exasim-project/NeoFOAM/tree/main/applications/solver/minimal>`_ example as we are still working on the basic implementations. But
we will gradually add example applications as we move along. Most likely we will initially work on porting solver applications, since pre- and postprocessing tools that work on a file-level basis are trivially interoperable.


.. toctree::
   :maxdepth: 2
   :caption: Contents:
Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
