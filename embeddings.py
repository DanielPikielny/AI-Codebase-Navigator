import ast
import asyncio
import logging
import os
import re
import time
from openai import AsyncOpenAI, OpenAI, RateLimitError, APIStatusError

logger = logging.getLogger(__name__)

async_client = AsyncOpenAI()
sync_client  = OpenAI()

CHUNK_SIZE   = 500
CHUNK_HARD   = 1200
EMBED_BATCH  = 50
EMBED_MODEL  = "text-embedding-3-small"
MAX_RETRIES  = 5
CONCURRENCY  = 5

SUPPORTED_EXTENSIONS = {
    ".py":   "python",
    ".js":   "javascript",
    ".ts":   "typescript",
    ".jsx":  "javascript",
    ".tsx":  "typescript",
    ".cpp":  "cpp",
    ".cc":   "cpp",
    ".cxx":  "cpp",
    ".hpp":  "cpp",
    ".h":    "cpp",
    ".java": "java",
}

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "dist", "build", ".next", "target", ".idea", ".vscode",
}

def load_code_files(repo_path: str) -> list:
    """
    Walk repo_path and return (full_path, source_code, language) for every
    supported source file, skipping common non-source directories.
    """
    results = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            ext  = os.path.splitext(fname)[1].lower()
            lang = SUPPORTED_EXTENSIONS.get(ext)
            if not lang:
                continue
            full_path = os.path.join(root, fname)
            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    code = f.read()
                if code.strip():
                    results.append((full_path, code, lang))
            except Exception:
                pass
    return results

def _split_large(text: str, max_size: int = CHUNK_HARD) -> list:
    """
    Split an oversized block by blank lines, then by raw chars if needed.
    FIX-⑥: max_size is now threaded through all recursive paths correctly.
    """
    if len(text) <= max_size:
        return [text]

    parts = re.split(r"\n{2,}", text)
    out, current = [], ""
    for part in parts:
        if len(current) + len(part) + 2 <= max_size:
            current = (current + "\n\n" + part).lstrip()
        else:
            if current:
                out.append(current)
            if len(part) > max_size:
                out.extend(part[i:i + max_size] for i in range(0, len(part), max_size))
                current = ""
            else:
                current = part
    if current:
        out.append(current)
    return [s for s in out if s.strip()]

def _chunk_python(code: str) -> list:
    """
    Split Python source by top-level function/class definitions using the AST.
    Methods inside classes are kept together with their class.
    Module-level code (imports, constants, etc.) is collected as its own chunk.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return _split_large(code)

    lines = code.splitlines(keepends=True)

    top_level_spans = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            top_level_spans.append((node.lineno - 1, node.end_lineno))

    top_level_spans.sort()
    covered = set()
    chunks  = []

    for start, end in top_level_spans:
        block = "".join(lines[start:end])
        chunks.extend(_split_large(block))
        covered.update(range(start, end))

    remainder = "".join(
        line for i, line in enumerate(lines) if i not in covered
    ).strip()
    if remainder:
        chunks.extend(_split_large(remainder))

    return [c for c in chunks if c.strip()]


_JS_TS_BOUNDARY = re.compile(
    r"^(?:"
    r"(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+\w+"
    r"|(?:export\s+)?(?:abstract\s+)?class\s+\w+"
    r"|(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?"
    r"(?:\(.*?\)|.*?)\s*=>"
    r")",
    re.MULTILINE,
)

_JAVA_CPP_BOUNDARY = re.compile(
    r"^[ \t]*(?:"
    r"(?:public|private|protected|static|final|abstract|override|virtual|inline"
    r"|explicit|constexpr|[[nodiscard]]\s+)+"
    r")?[\w:<>*&\[\]]+\s+\w+\s*\(",
    re.MULTILINE,
)

_CLASS_BOUNDARY = re.compile(
    r"^(?:(?:public|private|abstract|final|data|sealed)\s+)*class\s+\w+",
    re.MULTILINE,
)

def _chunk_by_regex(code: str, lang: str) -> list:
    if lang in ("javascript", "typescript"):
        pattern = _JS_TS_BOUNDARY
    else:
        pattern = re.compile(
            f"(?:{_JAVA_CPP_BOUNDARY.pattern})|(?:{_CLASS_BOUNDARY.pattern})",
            re.MULTILINE,
        )

    boundaries = [m.start() for m in pattern.finditer(code)]
    if not boundaries:
        return _split_large(code)

    segments = []
    prev = 0
    for boundary in boundaries:
        if boundary > prev:
            segments.append(code[prev:boundary])
        prev = boundary
    segments.append(code[prev:])

    chunks = []
    for seg in segments:
        chunks.extend(_split_large(seg.strip()))
    return [c for c in chunks if c.strip()]

def chunk_code(code: str, lang: str) -> list:
    if lang == "python":
        return _chunk_python(code)
    return _chunk_by_regex(code, lang)

def create_chunks(repo_path: str) -> list:
    """
    Load all supported source files from repo_path and return a flat list of
    chunk dicts: {"text": str, "source": str, "language": str}.
    """
    files  = load_code_files(repo_path)
    chunks = []
    for path, code, lang in files:
        for chunk in chunk_code(code, lang):
            chunks.append({"text": chunk, "source": path, "language": lang})
    return chunks

async def _embed_batch_with_retry(texts: list) -> list:
    """
    FIX-⑦: Embed one batch with exponential backoff on rate-limit and server errors.
    Retries up to MAX_RETRIES times before raising.
    """
    delay = 1.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await async_client.embeddings.create(
                model=EMBED_MODEL, input=texts
            )
            return [
                item.embedding
                for item in sorted(response.data, key=lambda x: x.index)
            ]
        except RateLimitError:
            if attempt == MAX_RETRIES:
                raise
            wait = delay * (2 ** (attempt - 1))
            logger.warning(
                "Rate limit hit (attempt %d/%d). Retrying in %.1fs…",
                attempt, MAX_RETRIES, wait,
            )
            await asyncio.sleep(wait)
        except APIStatusError as exc:
            if exc.status_code < 500 or attempt == MAX_RETRIES:
                raise
            wait = delay * (2 ** (attempt - 1))
            logger.warning(
                "API error %d (attempt %d/%d). Retrying in %.1fs…",
                exc.status_code, attempt, MAX_RETRIES, wait,
            )
            await asyncio.sleep(wait)
    raise RuntimeError("Embedding failed after all retries")

async def _embed_all_async(texts: list) -> list:
    """
    FIX-②: Semaphore limits concurrent requests to CONCURRENCY (default 5).
    This keeps us within Tier 1/2 RPM limits even on large repos.
    """
    batches   = [texts[i:i + EMBED_BATCH] for i in range(0, len(texts), EMBED_BATCH)]
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def _throttled(batch):
        async with semaphore:
            return await _embed_batch_with_retry(batch)

    results = await asyncio.gather(*[_throttled(b) for b in batches])
    return [emb for batch_result in results for emb in batch_result]

def embed_chunks(chunks: list) -> list:
    """Embed all chunks. Runs the async pipeline in a fresh event loop."""
    return asyncio.run(_embed_all_async([c["text"] for c in chunks]))

def embed_text(text: str) -> list:
    """
    FIX-⑧: Embed a single string synchronously.
    Avoids the overhead of spinning up an async event loop for one API call.
    """
    response = sync_client.embeddings.create(model=EMBED_MODEL, input=[text])
    return response.data[0].embedding