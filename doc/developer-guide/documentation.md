# Documentation Setup

The documentation for Ginkgo is set up using

- [Sphinx](https://www.sphinx-doc.org/)
- [Doxygen](https://www.doxygen.nl/)

Doxygen is used *only* for documenting the API through comments in the source code.
No explicit Doxygen file (i.e. a file with only Doxygen content) shall be written.
The `@cmd` syntax is used, where `cmd` is any valid doxygen command, see the list [here](https://www.doxygen.nl/manual/commands.html) for reference.

Any other documentation is done via Sphinx.
This includes the tutorial, how-to, and explanation aspects of the [diataxis](https://diataxis.fr/) approach.
The [MyST](https://myst-parser.readthedocs.io) extension is used to allow markdown files as Sphinx input.
Only markdown files shall be written for the Sphinx documentation.
Admonitions shall be added with the `:::` syntax. 
The triple \` is reserved for code blocks.

A connection from Sphinx to doxygen is established via [doxylink](https://github.com/sphinx-contrib/doxylink).
With this, it is possible to reference the doxygen documentation by using the syntax 
```md
{doxy}`gko::matrix::Dense`
```
Example: {doxy}`gko::matrix::Dense`.
In the other direction, only the main page for the Sphinx documentation is available through doxygen.
It is directly added to the `DoxygenLayout.xml` as a relative link.

The connections between Sphinx and doxygen rely on correctly set output directories for both.
The Sphinx output dir is the main one, and the doxygen output dir is defined relative to that.
The Sphinx output dir is set to `SPHINX_OUTPUT_DIR=$CMAKE_BINARY_DIR/doc/html` and the doxygen output *has* to be put under `SPHINX_OUTPUT_DIR/_doxygen`.
There will be an automatically generated subdir `html` of `_doxygen`.
As the documentation setup uses relative path between doxygen and sphinx, messing up the path will lead to broken references.