import os
import sys
sys.path.insert(0, os.path.abspath('../Churn_predict'))


# -- Project information -----------------------------------------------------

project = 'Churn_prediction'
authors = 'Yazid, Sivaprasad, Sanjaya'


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode'
]


# paths relative to this directory.
templates_path = ['_templates']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
