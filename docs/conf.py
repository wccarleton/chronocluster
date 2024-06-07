# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ChronoCluster'
copyright = '2024, W. Christopher Carleton'
author = 'W. Christopher Carleton'
release = '0.1'
version = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',        # Core autodoc extension
    'sphinx.ext.napoleon',       # Support for Google and NumPy style docstrings
    'sphinx.ext.viewcode',       # Link to the source code
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

# -- Autodoc configuration ---------------------------------------------------
# Define which members are documented
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': True,
    'show-inheritance': True,
}

# -- Napoleon settings -------------------------------------------------------
# Enable support for Google and NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Path setup --------------------------------------------------------------
# If your documentation needs a minimal Sphinx version, state it.
# needs_sphinx = '1.0'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
