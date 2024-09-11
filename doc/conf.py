# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Ginkgo'
copyright = '2024, The Ginkgo Authors'
author = 'The Ginkgo Authors'
release = '1.9.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx_sitemap',
    'sphinx.ext.inheritance_diagram',
    'sphinxcontrib.doxylink',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '_doxygen', 'joss']

highlight_language = 'c++'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_theme_options = {}
html_static_path = ['_static']
html_logo = '../assets/logo_doc.png'
html_favicon = '../assets/favicon.ico'
html_title = f'{project} v{release}'

# Define the canonical URL if you are using a custom domain on Read the Docs
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")

read_the_docs_build = os.environ.get('READTHEDOCS', None) == 'True'

# Tell Jinja2 templates the build is running on Read the Docs
if read_the_docs_build:
    if "html_context" not in globals():
        html_context = {}
    html_context["READTHEDOCS"] = True


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
myst_heading_anchors = 3

# -- doxylink configuration -------------------------------------------------
# https://sphinxcontrib-doxylink.readthedocs.io/en/stable/#

if read_the_docs_build:
    doxygen_dir = os.environ['READTHEDOCS_OUTPUT']
    doxylink = {
        'doxy': (f'{doxygen_dir}/html/_doxygen/Ginkgo.tag', f'{doxygen_dir}/html/_doxygen/usr')
    }
else:
    doxylink = {
        'doxy': ('_doxygen/Ginkgo.tag', '../_doxygen/usr')
    }
