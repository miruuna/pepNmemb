from __future__ import annotations

import pepNmemb

project = "pepNmemb"
author = "Miruna Serian"
year = "2025"
copyright = f"{year}, {author}"
# version = release = importlib.metadata.version("package")
version = f"v{pepNmemb.__version__}"
release = version
source_suffix = ".rst"
master_doc = "index"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.imgconverter",
]

autosummary_generate = True
autodoc_typehints = "signature"
autodoc_docstring_signature = True
autoclass_content = "both"

# pygments_style = 'trac'
pygmants_style = "default"
templates_path = ["."]
extlinks = {
    "pr": ("https://github.com/miruuna/pepNmemb/pull/%s", "PR #"),
    "mda": ("https://www.mdanalysis.org", "MDAnalysis"),
}

html_theme = "sphinx_rtd_theme"
html_use_smartypants = True
html_last_updated_fmt = "%b %d, %Y"
html_split_index = False
html_sidebars = {"**": ["searchbox.html", "globaltoc.html", "sourcelink.html"]}
html_short_title = f"{project}-{version}"

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False
