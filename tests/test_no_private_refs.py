"""Tripwire: shipped files must not carry home-directory paths, wiki links, unlisted email
addresses or listed terms.

The built-in checks are generic. A terms file adds more: one regular expression per line,
case-insensitive, `#` lines are comments. A line starting `prose:` applies only to prose: text
files, comments, string literals and JSON strings that contain a space, and Julia docstrings.
Code identifiers and data keys are therefore not flagged. A line starting `docs:` applies only
to text files. The terms file is `release/private_terms.txt` when present, or `--extra FILE`:

    python tests/test_no_private_refs.py [ROOT] [--extra FILE]

prints the hits and exits 1 on any. In a checkout with `release/public_paths.txt` the scanned
files are the ones that list selects, with `release/public/` replacing or adding to them;
anywhere else, every file.
"""
import io
import os
import re
import sys
import tokenize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIST = os.path.join("release", "public_paths.txt")
TERMS = os.path.join("release", "private_terms.txt")
OVERLAY = "release/public/"
SUFFIXES = {".py", ".md", ".toml", ".cff", ".txt", ".rst", ".cfg", ".yaml", ".yml", ".json",
            ".sh", ".jl", ".tsv", ".in", ""}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "build", "dist", "_build"}
PROSE_FILES = {".md", ".rst", ".cff", ".txt", ""}            # every line is prose
HASH_COMMENTS = {".sh", ".jl", ".toml", ".yaml", ".yml", ".cfg", ".in"}  # prose after "#"

GENERIC = [
    r"\[\[[A-Za-z][A-Za-z0-9_./ -]*(?:\|[^\]]*)?\]\]",       # wiki links, not numeric lists
    r"(?<![\w.])/(?:Users|home)/\w|[A-Za-z]:\\+Users\\+\w",  # absolute home directories
]
TOML_TABLE = re.compile(r"^\s*\[\[[\w.-]+\]\]\s*$")           # a TOML array-of-tables header

# Email addresses: only the project contact and GitHub noreply addresses may appear.
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b(?!\()")
ALLOWED_EMAILS = {"adityatandon29@gmail.com"}
NOREPLY = "@users.noreply.github.com"
STRING = re.compile(r'"((?:[^"\\]|\\.)*)"')                    # a double-quoted literal


def bad_email(line):
    return any(m.group(0).lower() not in ALLOWED_EMAILS and not m.group(0).lower().endswith(NOREPLY)
               for m in EMAIL.finditer(line))


def load_terms(path):
    """Regex lists from a terms file: {"": anywhere, "prose": prose only, "docs": text files}."""
    out = {"": [], "prose": [], "docs": []}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        kind, sep, rest = line.partition(":")
        if sep and kind in ("prose", "docs"):
            out[kind].append(rest.strip())
        else:
            out[""].append(line)
    return out


def patterns(extra=None):
    """Compiled (anywhere, prose-only, docs-only) patterns: the generic ones plus the terms file."""
    terms = load_terms(extra) if extra else {"": [], "prose": [], "docs": []}
    join = lambda ps: re.compile("|".join(f"(?:{p})" for p in ps), re.IGNORECASE) if ps else None
    return join(GENERIC + terms[""]), join(terms["prose"]), join(terms["docs"])


def default_extra(root):
    path = os.path.join(root, TERMS)
    return path if os.path.isfile(path) else None


def rules(root):
    """(include?, path, is_dir) in file order, or None when every file ships."""
    path = os.path.join(root, LIST)
    if not os.path.isfile(path):
        return None
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        keep = not line.startswith("!")
        line = line.lstrip("!")
        out.append((keep, line.rstrip("/"), line.endswith("/")))
    return out


def selected(rel, rs):
    """Whether repository-relative path `rel` ships under rules `rs` (last match wins)."""
    hit = False
    for keep, p, is_dir in rs:
        if rel == p or (is_dir and rel.startswith(p + "/")):
            hit = keep
    return hit


def shipped(root=ROOT):
    rs = rules(root)
    for dirpath, dirnames, names in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(names):
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            if os.path.splitext(name)[1] not in SUFFIXES:
                continue
            if rs is None or rel.startswith(OVERLAY):
                yield path
            elif selected(rel, rs) and not os.path.exists(os.path.join(root, OVERLAY, rel)):
                yield path                                     # an overlay copy replaces it


def py_prose(text):
    """{line number: prose} of Python source: comments, and string literals with a space."""
    out = {}
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return {n: line for n, line in enumerate(text.splitlines(), 1)}   # unparsable: all prose
    strings = {tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", tokenize.STRING)}
    for t in toks:
        if t.type == tokenize.COMMENT or (t.type in strings and re.search(r"\s", t.string)):
            for k, line in enumerate(t.string.splitlines(), t.start[0]):
                out[k] = out.get(k, "") + " " + line
    return out


def jl_prose(lines):
    """{line number: prose} of Julia source: comments, docstrings, strings with a space."""
    out, block = {}, None                                      # block: closing delimiter
    for n, line in enumerate(lines, 1):
        if block:
            out[n] = line
            if block in line:
                block = None
            continue
        for opener, closer in (('"""', '"""'), ("#=", "=#")):
            i = line.find(opener)
            if i >= 0 and closer not in line[i + len(opener):]:
                out[n], block = line, closer
        if n not in out:
            parts = [m.group(1) for m in STRING.finditer(line) if re.search(r"\s", m.group(1))]
            if "#" in line:
                parts.append(line.split("#", 1)[1])
            if parts:
                out[n] = " ".join(parts)
    return out


def hits(path, pats=None):
    anywhere, prose, docs = pats or patterns(default_extra(ROOT))
    ext = os.path.splitext(path)[1]
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    lines = text.splitlines()
    if ext == ".py":
        prose_of = py_prose(text)
    elif ext == ".jl":
        prose_of = jl_prose(lines)
    elif ext == ".json":
        prose_of = {n: " ".join(v for v in STRING.findall(line) if re.search(r"\s", v))
                    for n, line in enumerate(lines, 1)}
    elif ext in PROSE_FILES:
        prose_of = dict(enumerate(lines, 1))
    elif ext in HASH_COMMENTS:
        prose_of = {n: line.split("#", 1)[1] for n, line in enumerate(lines, 1) if "#" in line}
    else:
        prose_of = {}
    out = []
    for n, line in enumerate(lines, 1):
        bad = anywhere.search(line) and not (ext == ".toml" and TOML_TABLE.match(line))
        bad = bad or bad_email(line)
        bad = bad or (prose is not None and n in prose_of and prose.search(prose_of[n]))
        bad = bad or (docs is not None and ext in PROSE_FILES and docs.search(line))
        if bad:
            out.append(f"{n}: {line.strip()[:100]}")
    return out


def scan(root, extra):
    pats = patterns(extra)
    files = list(shipped(root))
    return files, [(os.path.relpath(p, root), h) for p in files for h in hits(p, pats)]


def test_shipped_files_exist():
    names = {os.path.relpath(p, ROOT).replace(os.sep, "/") for p in shipped()}
    assert {"pyproject.toml", "tnkit/__init__.py", "bench/suite/run.py",
            "tests/test_no_private_refs.py"} <= names


def test_no_private_references():
    _, bad = scan(ROOT, default_extra(ROOT))
    assert not bad, "private references in shipped files:\n" + "\n".join(
        f"{p}:{h}" for p, h in bad)


if __name__ == "__main__":
    args = sys.argv[1:]
    extra = None
    if "--extra" in args:
        i = args.index("--extra")
        extra = os.path.abspath(args[i + 1])
        del args[i:i + 2]
    root = os.path.abspath(args[0]) if args else ROOT
    files, bad = scan(root, extra or default_extra(root))
    for p, h in bad:
        print(f"{p}:{h}")
    print(f"tripwire: {len(files)} files, {len(bad)} hits")
    sys.exit(1 if bad or not files else 0)
