from __future__ import annotations
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from openai import OpenAI
from embeddings import SUPPORTED_EXTENSIONS, embed_text
from retriever import CodeRetriever

client  = OpenAI()
logger  = logging.getLogger(__name__)
PROMPTS = Path(__file__).parent / "prompts"

def _read_prompt(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8")

def _llm(prompt: str, max_tokens: int = 1500) -> str:
    response = client.chat.completions.create(
        model    = "gpt-4o-mini",
        messages = [{"role": "user", "content": prompt}],
        max_tokens = max_tokens,
    )
    return response.choices[0].message.content.strip()

def _context_from_results(results: list) -> str:
    return "\n\n".join(
        f"File: {r['source']}\n```\n{r['text']}\n```" for r in results
    )

def _safe_json_load(text: str, expected_type: type, fallback: Any) -> Any:
    """
    Strip markdown fences, parse JSON, validate type.
    Returns *fallback* on any parse error or type mismatch — never raises.

    Usage
        sub_qs = _safe_json_load(raw, list, [original_question])
    """
    try:
        clean = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
        data  = json.loads(clean)
        if isinstance(data, expected_type):
            return data
        logger.warning(
            "_safe_json_load: expected %s, got %s — using fallback",
            expected_type.__name__, type(data).__name__,
        )
    except json.JSONDecodeError as exc:
        logger.warning("_safe_json_load: JSON parse error — %s", exc)
    return fallback

@dataclass
class SubAnswer:
    question: str
    answer:   str
    sources:  list = field(default_factory=list)

@dataclass
class AgentResult:
    original_question: str
    sub_answers:       list
    final_answer:      str
    all_sources:       list = field(default_factory=list)

class Confidence:
    HIGH   = "High"
    MEDIUM = "Medium"
    LOW    = "Low"

    _PATTERN = re.compile(r"CONFIDENCE:\s*(High|Medium|Low)", re.IGNORECASE)

    @classmethod
    def parse(cls, text: str) -> str:
        """
        Extract the CONFIDENCE:<level> tag from a locate_prompt response.
        Returns Confidence.LOW if the tag is absent or malformed (safe default).
        """
        m = cls._PATTERN.search(text)
        if not m:
            logger.warning("Confidence tag not found in locate response; defaulting to Low")
            return cls.LOW
        raw = m.group(1).capitalize()
        return raw if raw in (cls.HIGH, cls.MEDIUM, cls.LOW) else cls.LOW


@dataclass
class LocateResult:
    query:      str
    answer:     str
    sources:    list
    confidence: str = Confidence.LOW 
def ask_multistep(
    retriever:        CodeRetriever,
    question:         str,
    mode_instruction: str,
    qa_template:      str,
    progress_fn=None,
    retrieve_fn=None,
) -> AgentResult:
    """
    Agent loop:
      1. Decompose the question into sub-questions  (FIX-B: safe JSON parse)
      2. For each sub-question: retrieve → answer
      3. Synthesize using both summaries AND raw evidence  (FIX-C)

    retrieve_fn: optional callable(query, k) → list[dict]
        Defaults to semantic-only search.  Pass the hybrid searcher from app_v4.
    """
    def _log(msg: str):
        if progress_fn:
            progress_fn(msg)

    def _retrieve(q: str, k: int = 4) -> list:
        if retrieve_fn is not None:
            return retrieve_fn(q, k)
        return retriever.search(embed_text(q), k=k)

    _log("Breaking question into sub-questions…")
    raw_decompose = _llm(
        _read_prompt("decompose_prompt.txt").format(question=question),
        max_tokens=300,
    )
    sub_questions: list = _safe_json_load(
        raw_decompose, list, fallback=[question]
    )
    sub_questions = [sq for sq in sub_questions if isinstance(sq, str) and sq.strip()]
    if not sub_questions:
        sub_questions = [question]
    sub_questions = sub_questions[:4]
    sub_answers:  list = []
    seen_sources: dict = {}
    all_raw_snippets: list = []  

    for i, sq in enumerate(sub_questions, 1):
        _log(f"Sub-question {i}/{len(sub_questions)}: {sq[:70]}…")
        results = _retrieve(sq)
        context = _context_from_results(results)
        prompt  = (
            qa_template.format(context=context, question=sq)
            + f"\n\nTone / depth: {mode_instruction}"
        )
        answer = _llm(prompt, max_tokens=600)
        sub_answers.append(SubAnswer(question=sq, answer=answer, sources=results))

        for r in results:
            seen_sources.setdefault(r["source"], r)
            all_raw_snippets.append(r)   # FIX-C

    _log("Synthesising final answer…")
    findings = "\n\n".join(
        f"### Sub-question {i}: {sa.question}\n{sa.answer}"
        for i, sa in enumerate(sub_answers, 1)
    )
    seen_evidence: set = set()
    unique_snippets: list = []
    for r in all_raw_snippets:
        key = (r["source"], r["text"][:120])
        if key not in seen_evidence:
            seen_evidence.add(key)
            unique_snippets.append(r)
    raw_evidence = _context_from_results(unique_snippets)

    synth_prompt = _read_prompt("synthesize_prompt.txt").format(
        question         = question,
        findings         = findings,
        raw_evidence     = raw_evidence, 
        mode_instruction = mode_instruction,
    )
    final = _llm(synth_prompt, max_tokens=1800)

    return AgentResult(
        original_question = question,
        sub_answers       = sub_answers,
        final_answer      = final,
        all_sources       = list(seen_sources.values()),
    )

def locate_feature(
    retriever:        CodeRetriever,
    query:            str,
    mode_instruction: str,
    k: int = 8,
    retrieve_fn=None,
) -> LocateResult:
    """
    Broad retrieval followed by LLM ranking.
    FIX-D: parses CONFIDENCE:<level> tag and exposes it as a structured field.
    """
    if retrieve_fn is not None:
        results = retrieve_fn(query, k)
    else:
        results = retriever.search(embed_text(query), k=k)

    context = _context_from_results(results)
    prompt  = _read_prompt("locate_prompt.txt").format(
        query            = query,
        context          = context,
        mode_instruction = mode_instruction,
    )
    answer     = _llm(prompt, max_tokens=900)
    confidence = Confidence.parse(answer) 

    return LocateResult(
        query      = query,
        answer     = answer,
        sources    = results,
        confidence = confidence,
    )

def suggest_refactor(
    file_path:        str,
    mode_instruction: str,
    max_chars: int = 6000,
) -> str:
    try:
        code = Path(file_path).read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Could not read file: {exc}"
    ext  = os.path.splitext(file_path)[1].lower()
    lang = SUPPORTED_EXTENSIONS.get(ext, "text")
    return _llm(
        _read_prompt("refactor_prompt.txt").format(
            file_path        = file_path,
            language         = lang,
            code             = code[:max_chars],
            mode_instruction = mode_instruction,
        ),
        max_tokens=2000,
    )

def generate_tests(
    file_path:        str,
    mode_instruction: str,
    max_chars: int = 6000,
) -> str:
    try:
        code = Path(file_path).read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Could not read file: {exc}"
    ext  = os.path.splitext(file_path)[1].lower()
    lang = SUPPORTED_EXTENSIONS.get(ext, "text")
    return _llm(
        _read_prompt("test_prompt.txt").format(
            file_path        = file_path,
            language         = lang,
            code             = code[:max_chars],
            mode_instruction = mode_instruction,
        ),
        max_tokens=2500,
    )

def explain_file(file_path: str, mode_instruction: str, max_chars: int = 6000) -> str:
    try:
        code = Path(file_path).read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Could not read file: {exc}"
    ext  = os.path.splitext(file_path)[1].lower()
    lang = SUPPORTED_EXTENSIONS.get(ext, "text")
    return _llm(f"""You are a senior software engineer explaining a source file.

File: {file_path}
Language: {lang}

```{lang}
{code[:max_chars]}
```

Provide a structured explanation:
1. **Purpose** — what this file does in 1–2 sentences.
2. **Key functions / classes** — each with a brief description.
3. **Dependencies** — external libraries and internal imports, and why they're used.
4. **Notable patterns** — design patterns, idioms, or non-obvious choices.

Tone / depth: {mode_instruction}
""", max_tokens=1200)

def summarize_repo(repo_path: str, mode_instruction: str) -> str:
    extensions = set(SUPPORTED_EXTENSIONS.keys())
    skip       = {
        "__pycache__", "node_modules", ".git", ".venv", "venv",
        "dist", "build", ".next", "target",
    }
    snippets: list = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in skip]
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext in extensions:
                fpath = os.path.join(root, fname)
                try:
                    content = Path(fpath).read_text(encoding="utf-8", errors="replace")[:1500]
                    lang    = SUPPORTED_EXTENSIONS[ext]
                    snippets.append(f"### {fpath}\n```{lang}\n{content}\n```")
                except Exception:
                    pass
            if len(snippets) >= 20:
                break
        if len(snippets) >= 20:
            break

    return _llm(f"""You are a senior software engineer summarising a codebase.

{chr(10).join(snippets)}

Provide a structured summary:
1. **Architecture** — high-level structure of the system.
2. **Main modules** — each file/module and its role.
3. **Data flow** — how data enters, transforms, and exits.
4. **Key dependencies** — important third-party libraries and what they do.
5. **Potential improvements** — 2–3 concrete, actionable suggestions.

Tone / depth: {mode_instruction}
""", max_tokens=2000)