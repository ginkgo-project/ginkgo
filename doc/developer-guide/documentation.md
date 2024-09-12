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
Examples:

- {doxy}`gko::matrix::Dense`: referencing a type
- {doxy}`gko::array::array`: referencing a function without specifying overload
- {doxy}`gko::array::array(std::shared_ptr< const Executor >, size_type, value_type *)`: reference a concrete function overload

The doxygen API documentation is embedded into the Sphinx documentation via [doxysphinx](https://boschglobal.github.io/doxysphinx/).
This setup is a bit clunky and introduces some constraints on the placement of the doxygen generated files.
The doxygen output directory has to be configured to be under the Sphinx source directory.
Since the Sphinx sources are part of the normal source tree, this results in those options:

1. Generate the generated doxygen files into the source tree
2. Copy the Sphinx sources into the build tree

We chose option 2., as it keeps the source tree clean.
Directories with Sphinx sources, and top-level `.md` files under `/doc` are added to the build tree as symbolic links.
If new directories or top-level files are added, new links have to be created manually (via CMake).
