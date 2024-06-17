# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Ginkgo'
copyright = '2024, The Ginkgo Authors'
author = 'The Ginkgo Authors'
release = '1.9.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['myst_parser', 'breathe']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']


# -- MyST configuration -------------------------------------------------------
# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "linkify",
    "replacements",
    "smartquotes"
]


# -- breathe configuration ---------------------------------------------------
# https://breathe.readthedocs.io/en/latest/quickstart.html

breathe_projects = {"ginkgo": "/home/marcel/projects/working-trees/ginkgo/user-guide/cmake-build-debug/doc/doxygen/usr/xml"}
breathe_default_project = "ginkgo"
breathe_default_members = ("members-only", "outline")