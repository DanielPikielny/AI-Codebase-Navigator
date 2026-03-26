from __future__ import annotations
import ast
import os
import re
from collections import defaultdict
from pathlib import Path
from embeddings import SUPPORTED_EXTENSIONS

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "dist", "build", ".next", "target", ".idea", ".vscode",
}

def _is_inside(child: Path, parent: Path) -> bool:
    """
    Return True if *child* is inside *parent*.
    Uses os.path.commonpath so it works on Python 3.7+ and handles
    case-insensitive file systems correctly.
    """
    try:
        child_str  = os.path.normcase(str(child.resolve()))
        parent_str = os.path.normcase(str(parent.resolve()))
        return os.path.commonpath([child_str, parent_str]) == parent_str
    except ValueError:
        return False

def _safe_id(text: str) -> str:
    """Replace any character that is not alphanumeric or underscore with '_'."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", text)

def _imports_python(code: str, this_path: Path, repo_root: Path) -> list:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    module_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_names.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            module_names.append(node.module.split(".")[0])

    resolved = []
    for mod in module_names:
        if not mod:
            continue
        candidate_file = repo_root / (mod.replace(".", os.sep) + ".py")
        candidate_pkg  = repo_root / mod.replace(".", os.sep) / "__init__.py"
        for c in (candidate_file, candidate_pkg):
            if c.is_file():
                resolved.append(c.resolve())
                break
    return resolved

_JS_IMPORT_RE = re.compile(
    r"""(?:import\s+.*?\s+from|require\s*\()\s*['"]([^'"]+)['"]""",
    re.MULTILINE,
)

def _imports_js_ts(code: str, this_path: Path, repo_root: Path) -> list:
    resolved  = []
    base_dir  = this_path.parent
    repo_str  = os.path.normcase(str(repo_root.resolve()))

    for m in _JS_IMPORT_RE.findall(code):
        if not m.startswith("."):
            continue
        raw = (base_dir / m).absolute()
        for ext in ("", ".js", ".ts", ".jsx", ".tsx", "/index.js", "/index.ts"):
            candidate = Path(str(raw) + ext)
            try:
                cand_str = os.path.normcase(str(candidate.resolve()))
                if (
                    candidate.is_file()
                    and os.path.commonpath([cand_str, repo_str]) == repo_str
                ):
                    resolved.append(candidate.resolve())
                    break
            except (OSError, ValueError):
                continue
    return resolved

_JAVA_IMPORT_RE = re.compile(r"^import\s+([\w.]+);", re.MULTILINE)

def _imports_java(code: str, this_path: Path, repo_root: Path) -> list:
    resolved = []
    for m in _JAVA_IMPORT_RE.findall(code):
        parts     = m.split(".")
        candidate = repo_root / Path(*parts[:-1]) / (parts[-1] + ".java")
        if candidate.is_file():
            resolved.append(candidate.resolve())
    return resolved

_CPP_INCLUDE_RE = re.compile(r'^#include\s+"([^"]+)"', re.MULTILINE)

def _imports_cpp(code: str, this_path: Path, repo_root: Path) -> list:
    resolved = []
    for m in _CPP_INCLUDE_RE.findall(code):
        candidate = (this_path.parent / m).resolve()
        if candidate.is_file() and _is_inside(candidate, repo_root):
            resolved.append(candidate)
    return resolved

_EXTRACTORS = {
    "python":     _imports_python,
    "javascript": _imports_js_ts,
    "typescript": _imports_js_ts,
    "java":       _imports_java,
    "cpp":        _imports_cpp,
}

def build_dependency_graph(repo_path: str):
    """
    Walk *repo_path*, analyse imports, and return:
      adjacency : dict[rel_path_str -> list[rel_path_str]]
      dot       : graphviz.Digraph

    Raises ImportError if the graphviz Python package is missing.
    """
    try:
        import graphviz
    except ImportError:
        raise ImportError(
            "graphviz Python package not installed.\n"
            "Run: pip install graphviz\n"
            "You also need the Graphviz binary:\n"
            "  macOS  : brew install graphviz\n"
            "  Linux  : sudo apt install graphviz\n"
            "  Windows: https://graphviz.org/download/"
        )

    repo_root  = Path(repo_path).resolve()
    extensions = set(SUPPORTED_EXTENSIONS.keys())

    all_files = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext in extensions:
                all_files.append(Path(os.path.join(root, fname)).resolve())

    file_set = set(all_files)
    adjacency = {}
    for fpath in all_files:
        rel       = str(fpath.relative_to(repo_root))
        ext       = fpath.suffix.lower()
        lang      = SUPPORTED_EXTENSIONS.get(ext, "")
        extractor = _EXTRACTORS.get(lang)
        adjacency[rel] = []

        if not extractor:
            continue
        try:
            code = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        for dep in extractor(code, fpath, repo_root):
            if dep in file_set:
                dep_rel = str(dep.relative_to(repo_root))
                if dep_rel != rel and dep_rel not in adjacency[rel]:
                    adjacency[rel].append(dep_rel)

    dot = graphviz.Digraph(name="dependencies")
    dot.attr(
        rankdir  = "LR",
        bgcolor  = "transparent",
        fontname = "Helvetica",
        splines  = "ortho",
        nodesep  = "0.5",
        ranksep  = "0.8",
        overlap  = "false",
    )
    dot.attr(
        "node",
        shape     = "box",
        style     = "rounded,filled",
        fillcolor = "#1c2d3f",
        fontcolor = "#79c0ff",
        color     = "#388bfd",
        fontname  = "Helvetica",
        fontsize  = "11",
    )
    dot.attr("edge", color="#555555", arrowsize="0.7")

    node_id = {rel: f"n{i}" for i, rel in enumerate(adjacency)}

    by_dir = defaultdict(list)
    for rel in adjacency:
        folder = str(Path(rel).parent)
        by_dir[folder].append(rel)

    for folder, rels in sorted(by_dir.items()):
        label      = folder if folder != "." else "(root)"
        cluster_id = _safe_id(f"cluster_{folder}")  
        with dot.subgraph(name=cluster_id) as sg:
            sg.attr(
                label     = label,
                style     = "rounded",
                color     = "#30363d",
                fontcolor = "#8b949e",
                fontname  = "Helvetica",
                fontsize  = "10",
            )
            for rel in rels:
                sg.node(node_id[rel], label=Path(rel).name, tooltip=rel)

    for rel, deps in adjacency.items():
        for dep in deps:
            if dep in node_id:
                dot.edge(node_id[rel], node_id[dep])

    return adjacency, dot