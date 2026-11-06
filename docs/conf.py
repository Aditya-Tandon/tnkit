"""Sphinx configuration for the tnkit documentation."""
from __future__ import annotations

import inspect
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import tnkit  # noqa: E402

project = "tnkit"
author = "Aditya Tandon"
copyright = "2026, Aditya Tandon"
release = tnkit.__version__
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_nb",                 # Markdown (MyST) and notebooks; activates myst_parser itself
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
]

exclude_patterns = ["_build", "_templates", "BUILD.md", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]

# --- MyST and notebooks --------------------------------------------------------------
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 3
nb_execution_mode = "off"      # notebooks render their stored outputs; nothing runs on build

# --- API reference -------------------------------------------------------------------
autosummary_generate = True
autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
napoleon_numpy_docstring = True
napoleon_google_docstring = True
napoleon_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
}

# --- Maths: bra-ket and tensor-index macros, usable in Markdown and docstrings --------
mathjax3_config = {
    "tex": {
        "macros": {
            "ket": [r"\left|#1\right\rangle", 1],
            "bra": [r"\left\langle #1\right|", 1],
            "braket": [r"\left\langle #1\middle|#2\right\rangle", 2],
            "ketbra": [r"\left|#1\right\rangle\!\left\langle #2\right|", 2],
            "expval": [r"\left\langle #1\right\rangle", 1],
            "Tr": r"\operatorname{Tr}",
            # MPS tensor A^{s}_{ab} and MPO tensor W^{s s'}_{ab}
            "mps": [r"{#1}^{#2}_{#3 #4}", 4],
            "mpo": [r"{#1}^{#2 #3}_{#4 #5}", 5],
        }
    }
}

# --- HTML ----------------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = f"tnkit {release}"
html_theme_options = {
    "logo": {"text": "tnkit"},
    "github_url": "https://github.com/Aditya-Tandon/tnkit",
    "header_links_before_dropdown": 6,
    "navigation_with_keys": False,
    "show_toc_level": 2,
}
html_context = {"default_mode": "auto"}
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
copybutton_selector = "div:not(.highlight-text) > div.highlight pre"   # not verbatim docstrings


# --- Plain-text docstrings ----------------------------------------------------------
# Many tnkit docstrings are plain text rather than reST: free-form underlined headings,
# aligned option tables, |x| for norms. Rendered as reST they raise errors and lose their
# layout. After napoleon has run, each docstring is test-parsed: headings napoleon did not
# claim become rubrics, and a docstring that still does not parse cleanly is shown verbatim.
_UNDERLINE = re.compile(r"^\s*([-=~^])\1{2,}\s*$")
_ROLE = re.compile(r":[A-Za-z][\w:+.-]*:(?=`)")


def _headings_to_rubrics(lines):
    out, i = [], 0
    while i < len(lines):
        line, nxt = lines[i], lines[i + 1] if i + 1 < len(lines) else ""
        if line.strip() and not _UNDERLINE.match(line) and _UNDERLINE.match(nxt):
            indent = line[: len(line) - len(line.lstrip())]
            out += [f"{indent}.. rubric:: {line.strip()}", ""]
            i += 2
            continue
        out.append(line)
        i += 1
    return out


def _parses_cleanly(lines):
    import io
    from docutils.core import publish_doctree

    report = io.StringIO()
    try:
        publish_doctree(_ROLE.sub("", "\n".join(lines)), settings_overrides={
            "report_level": 2, "halt_level": 5, "warning_stream": report})
    except Exception:                       # a Sphinx-only construct; let Sphinx handle it
        return True
    return not report.getvalue()


def _render_docstring(app, what, name, obj, options, lines):
    fixed = _headings_to_rubrics(lines)
    if _parses_cleanly(fixed):
        lines[:] = fixed
        return
    raw = inspect.getdoc(obj) if what in ("module", "class", "exception", "function",
                                         "method") else None
    body = raw.splitlines() if raw else lines
    lines[:] = [".. code-block:: text", ""] + [f"   {ln}" if ln.strip() else "" for ln in body] + [""]


# --- API pages generated from the package --------------------------------------------
# api/public.rst lists tnkit.__all__, and each entry of _MODULE_PAGES gets a page of its
# own; all are written when a build starts, so the reference follows the code.
# Names are grouped by the module that defines them; a name from a module not listed lands
# under "Other".
_GROUPS = [
    ("Ground states", ("tnkit._dmrg", "tnkit.dmrg_excited", "tnkit._idmrg", "tnkit._vumps")),
    ("Time evolution", ("tnkit.tdvp", "tnkit.tebd", "tnkit.batch_tdvp")),
    ("States", ("tnkit.mps",)),
    ("Models", ("tnkit.models", "tnkit.models_lgt")),
    ("Observables", ("tnkit.observables",)),
    ("Backends", ("tnkit.backend",)),
    ("Block-sparse tensors and symmetries (provisional)",
     ("tnkit.blocks", "tnkit.block_backend", "tnkit.symmetry")),
]
_NAME_GROUP = {"random_mps": "States", "right_canonicalize": "States",
               "mpo_expectation": "Observables", "mpo_variance": "Observables",
               "two_site_variance": "Observables"}
_DATA_DOCS = {"SX": "Pauli matrix σˣ, a 2×2 complex numpy array.",
              "SY": "Pauli matrix σʸ, a 2×2 complex numpy array.",
              "SZ": "Pauli matrix σᶻ, a 2×2 complex numpy array."}
_PROVISIONAL = (".. warning::", "",
                "   Provisional: the block-tensor modules, including the ``block_graph`` and "
                "``block_lanczos`` switches, may change before 1.0.", "")
# (file name, title, modules, provisional)
_MODULE_PAGES = [
    ("blocks", "Block-sparse tensors",
     ("tnkit.blocks", "tnkit.block_backend", "tnkit.block_graph", "tnkit.block_lanczos"), True),
    ("opsum", "Operator sums", ("tnkit.opsum",), False),
]


def _heading(text, char):
    return [text, char * len(text), ""]


def _table(names):
    return [".. autosummary::", "   :toctree: generated", ""] + [f"   {n}" for n in names] + [""]


def _defined_in(mod):
    """A module's public functions and classes: its __all__, else what it defines itself."""
    names = getattr(mod, "__all__", None)
    if names is None:
        names = [n for n, v in vars(mod).items() if not n.startswith("_")
                 and (inspect.isfunction(v) or inspect.isclass(v))
                 and getattr(v, "__module__", None) == mod.__name__]
    return list(names)


def _write(name, lines):
    """Write an API page only when its content changes, so a rebuild leaves it alone."""
    path, text = os.path.join(HERE, "api", f"{name}.rst"), "\n".join(lines)
    try:
        with open(path, encoding="utf-8") as fh:
            if fh.read() == text:
                return
    except OSError:
        pass
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _write_api_pages(app=None, config=None):
    import importlib

    groups = {title: [] for title, _ in _GROUPS}
    groups["Other"], data = [], []
    for name in tnkit.__all__:
        obj = getattr(tnkit, name)
        if inspect.ismodule(obj) or name.startswith("__"):
            continue
        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            data.append(name)
            continue
        mod = getattr(obj, "__module__", "")
        title = _NAME_GROUP.get(name) or next((t for t, m in _GROUPS if mod in m), "Other")
        groups[title].append(name)

    lines = _heading("Public API", "=")
    lines += ["Everything on this page is importable from ``tnkit`` directly.", "",
              ".. currentmodule:: tnkit", ""]
    for title, names in groups.items():
        if names:
            lines += _heading(title, "-")
            if "provisional" in title:
                lines += list(_PROVISIONAL)
            lines += _table(names)
    if data:
        lines += _heading("Constants", "-")
        for n in data:
            lines += [f".. py:data:: {n}", "", f"   {_DATA_DOCS.get(n, 'Exported constant.')}", ""]
    _write("public", lines)

    for page, title, modnames, provisional in _MODULE_PAGES:
        lines = _heading(title, "=") + (list(_PROVISIONAL) if provisional else [])
        for modname in modnames:
            mod = importlib.import_module(modname)
            lines += _heading(f"``{modname}``", "-")
            lines += [f".. automodule:: {modname}", "   :no-members:", "",
                      f".. currentmodule:: {modname}", ""] + _table(_defined_in(mod))
        _write(page, lines)


def setup(app):
    # At config-inited, not builder-inited: Sphinx lists the source files before
    # builder-inited, and a page written after that is missed on a fresh build.
    app.connect("config-inited", _write_api_pages)
    app.connect("autodoc-process-docstring", _render_docstring)


if __name__ == "__main__":      # write the API pages without a build: python docs/conf.py
    _write_api_pages()
