.. _basics_unstructuredMesh:

unstructuredMesh
================

The `unstructuredMesh` in the current implementation stores the relevant data for the `unstructuredMesh` on the selected executor. So it is currently a data container for mesh data.

.. warning::
   - unable to read the mesh from disc
   - mesh data needs to be provided by the user

.. doxygenclass:: NeoFOAM::UnstructuredMesh
   :members:

BoundaryMesh
^^^^^^^^^^^^

The boundaryMesh in the current implementation stores the relevant data for the `boundaryMesh` on the selected executor. So it is currently a data container for boundary mesh data. The `boundaryMesh` information are stored in a continuous array and the index for the boundary patch is stored in a offset.

.. warning::
   - unable to read the boundary mesh from disc
   - boundary mesh data needs to be provided by the user

.. doxygenclass:: NeoFOAM::BoundaryMesh
   :members:

StencilDataBase
^^^^^^^^^^^^^^^

Offers the ability to register additional stencil data for the `unstructuredMesh`. If the mesh changes, the stencil data is automatically updated.

.. warning::
   `StencilDataBas` is currently a placeholder based on dictionary. The implementation is not yet complete and requires the additional of additional members functions that are linked to the implementation of the operators

.. warning::
   - unable to read the boundary mesh from disc
   - boundary mesh data needs to be provided by the user

.. doxygenclass:: NeoFOAM::StencilDataBase
   :members:
