# source https://raw.githubusercontent.com/deepmind/dm-haiku/main/docs/conf.py


# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

from __future__ import annotations
import doctest
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import sphinxcontrib.katex as katex

import pytreeclass

# -- Project information -----------------------------------------------------

project = "pytreeclass"
copyright = "2022, Mahmoud Asem"
author = "Mahmoud Asem"


# -- General configuration ---------------------------------------------------
master_doc = "index"


# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.katex",
    "sphinx_autodoc_typehints",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for autodoc -----------------------------------------------------

autodoc_default_options = {
    "member-order": "bysource",
    "special-members": True,
    "exclude-members": "__repr__, __str__, __weakref__",
}


# -- Options for HTML output -------------------------------------------------


# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]


# -- Options for doctest -----------------------------------------------------

doctest_test_doctest_blocks = "true"
doctest_global_setup = """
import jax
import jax.numpy as jnp
import pytreeclass
"""
doctest_default_flags = (
    doctest.ELLIPSIS
    | doctest.IGNORE_EXCEPTION_DETAIL
    | doctest.DONT_ACCEPT_TRUE_FOR_1
    | doctest.NORMALIZE_WHITESPACE
)


# -- Options for katex ------------------------------------------------------

# See: https://sphinxcontrib-katex.readthedocs.io/en/0.4.1/macros.html
latex_macros = r"""
    \def \d              #1{\operatorname{#1}}
"""

# Translate LaTeX macros to KaTeX and add to options for HTML builder
katex_macros = katex.latex_defs_to_katex_macros(latex_macros)
katex_options = "macros: {" + katex_macros + "}"

# Add LaTeX macros for LATEX builder
latex_elements = {"preamble": latex_macros}


# -- Source code links -------------------------------------------------------


def linkcode_resolve(domain, info):
    """Resolve a GitHub URL corresponding to Python object."""
    if domain != "py":
        return None

    try:
        mod = sys.modules[info["module"]]
    except ImportError:
        return None

    obj = mod
    try:
        for attr in info["fullname"].split("."):
            obj = getattr(obj, attr)
    except AttributeError:
        return None
    else:
        obj = inspect.unwrap(obj)

    try:
        filename = inspect.getsourcefile(obj)
    except TypeError:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        return None

    return "https://github.com/ASEM000/pytreeclass/%s#L%d#L%d" % (
        os.path.relpath(filename, start=os.path.dirname(pytreeclass.__file__)),
        lineno,
        lineno + len(source) - 1,
    )


# -- nbsphinx configuration --------------------------------------------------

nbsphinx_execute = "never"
nbsphinx_codecell_lexer = "ipython"
nbsphinx_kernel_name = "python"
nbsphinx_timeout = 180
nbsphinx_prolog = r"""
{% set docname = 'docs/' + env.doc2path(env.docname, base=None) %}
.. only:: html
    .. role:: raw-html(raw)
        :format: html
    .. nbinfo::
        Interactive online version:
        :raw-html:`<a href="https://colab.research.google.com/github/ASEM000/pytreeclass/blob/main/{{ docname }}"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg" style="vertical-align:text-bottom"></a>`
"""