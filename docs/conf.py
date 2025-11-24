"""Sphinx configuration for ReClaim documentation."""

import os
import sys
from datetime import datetime

# Ensure project source is discoverable by autodoc.
sys.path.insert(0, os.path.abspath("../src"))

project = "ReClaim"
author = "ReClaim contributors"
copyright = f"{datetime.now().year}, {author}"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
]
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_static_path = ["_static"]
html_theme = "alabaster"
