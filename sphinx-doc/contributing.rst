Contributing
^^^^^^^^^^^^

Contributions are highly welcomed. Here is some information getting you started.

Building the Documentation
""""""""""""""""""""""""""

NeoFOAMs documentation can be found `main <https://exasim-project.com/NeoFOAM/>`_  and `doxygen <https://exasim-project.com/NeoFOAM/doxygen/html/>`_ documentation can be found online. However, if you want to build the documentation locally you can do so, by executing the following steps.
First, make sure that Sphinx and Doxygen are installed on your system. Second, execute the following commands:

   .. code-block:: bash

    cmake -B build -DNEOFOAM_BUILD_DOC=ON # configure the build
    cmake --build build --target sphinx # build the documentation
    # or
    sphinx-build -b html ./doc ./docs_build


The documentation will be built in the `docs_build` directory and can be viewed by opening the `index.html` file in a web browser.

   .. code-block:: bash

    firefox docs_build/index.html

Alternatively, the documentation can be built by just adding the `-DNEOFOAM_BUILD_DOC=ON` to the configuration step of the build process and then building the documentation using the `sphinx` target.
