from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from openai import OpenAI

from embeddings import SUPPORTED_EXTENSIONS, embed_text
from retriever import CodeRetriever

client = OpenAI()
PROMPTS = Path(__file__).parent / "prompts"

def _read_prompt(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8")

def _llm(prompt: str, max_tokens: int = 1500) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()

def _context_from_results(results: list[dict]) -> str:
    return "\n\n".join(
        f"File: {r['source']}\n```\n{r['text']}\n```" for r in results
    )

@dataclass
class SubAnswer:
    question: str
    answer:   str
    sources:  list[dict] = field(default_factory=list)

@dataclass
class AgentResult:
    original_question: str
    sub_answers:       list[SubAnswer]
    final_answer:      str
    all_sources:       list[dict] = field(default_factory=list)

def ask_multistep(
    retriever:       CodeRetriever,
    question:        str,
    mode_instruction: str,
    qa_template:     str,
    progress_fn=None,
) -> AgentResult:
    """
    Agent loop:
      1. Decompose the question into sub-questions.
      2. For each sub-question: embed → retrieve → answer.
      3. Synthesize all sub-answers into one coherent final answer.
    """
    def _log(msg: str):
        if progress_fn:
            progress_fn(msg)

    _log("Breaking question into sub-questions…")
    decompose_prompt = _read_prompt("decompose_prompt.txt").format(question=question)
    raw = _llm(decompose_prompt, max_tokens=300)

    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        sub_questions: list[str] = json.loads(clean)
        if not isinstance(sub_questions, list):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        sub_questions = [question]

    sub_questions = sub_questions[:4]
    sub_answers: list[SubAnswer] = []
    seen_sources: dict[str, dict] = {}

    for i, sq in enumerate(sub_questions, 1):
        _log(f"Researching sub-question {i}/{len(sub_questions)}: {sq[:60]}…")
        emb     = embed_text(sq)
        results = retriever.search(emb, k=4)
        context = _context_from_results(results)
        prompt  = (
            qa_template.format(context=context, question=sq)
            + f"\n\nTone / depth: {mode_instruction}"
        )
        answer = _llm(prompt, max_tokens=600)
        sub_answers.append(SubAnswer(question=sq, answer=answer, sources=results))

        for r in results:
            seen_sources.setdefault(r["source"], r)

    _log("Synthesising final answer…")
    findings = "\n\n".join(
        f"### Sub-question {i}: {sa.question}\n{sa.answer}"
        for i, sa in enumerate(sub_answers, 1)
    )
    synth_prompt = _read_prompt("synthesize_prompt.txt").format(
        question         = question,
        findings         = findings,
        mode_instruction = mode_instruction,
    )
    final = _llm(synth_prompt, max_tokens=1500)

    return AgentResult(
        original_question = question,
        sub_answers       = sub_answers,
        final_answer      = final,
        all_sources       = list(seen_sources.values()),
    )

@dataclass
class LocateResult:
    query:    str
    answer:   str
    sources:  list[dict]

def locate_feature(
    retriever:        CodeRetriever,
    query:            str,
    mode_instruction: str,
    k: int = 8,
) -> LocateResult:
    """
    Broad retrieval (k=8) followed by LLM ranking and explanation of candidates.
    """
    emb     = embed_text(query)
    results = retriever.search(emb, k=k)
    context = _context_from_results(results)

    prompt = _read_prompt("locate_prompt.txt").format(
        query            = query,
        context          = context,
        mode_instruction = mode_instruction,
    )
    answer = _llm(prompt, max_tokens=800)
    return LocateResult(query=query, answer=answer, sources=results)

def suggest_refactor(
    file_path:        str,
    mode_instruction: str,
    max_chars: int = 6000,
) -> str:
    """
    Read *file_path* and return markdown-formatted refactoring suggestions
    with concrete code examples.
    """
    try:
        code = Path(file_path).read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Could not read file: {exc}"

    ext  = os.path.splitext(file_path)[1].lower()
    lang = SUPPORTED_EXTENSIONS.get(ext, "text")

    prompt = _read_prompt("refactor_prompt.txt").format(
        file_path        = file_path,
        language         = lang,
        code             = code[:max_chars],
        mode_instruction = mode_instruction,
    )
    return _llm(prompt, max_tokens=2000)

def generate_tests(
    file_path:        str,
    mode_instruction: str,
    max_chars: int = 6000,
) -> str:
    """
    Read *file_path* and return a complete runnable test file as a string.
    """
    try:
        code = Path(file_path).read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Could not read file: {exc}"

    ext  = os.path.splitext(file_path)[1].lower()
    lang = SUPPORTED_EXTENSIONS.get(ext, "text")

    prompt = _read_prompt("test_prompt.txt").format(
        file_path        = file_path,
        language         = lang,
        code             = code[:max_chars],
        mode_instruction = mode_instruction,
    )
    return _llm(prompt, max_tokens=2500)

def explain_file(file_path: str, mode_instruction: str, max_chars: int = 6000) -> str:
    try:
        code = Path(file_path).read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Could not read file: {exc}"

    ext  = os.path.splitext(file_path)[1].lower()
    lang = SUPPORTED_EXTENSIONS.get(ext, "text")

    prompt = f"""You are a senior software engineer explaining a source file.

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
"""
    return _llm(prompt, max_tokens=1200)

def summarize_repo(repo_path: str, mode_instruction: str) -> str:
    extensions = set(SUPPORTED_EXTENSIONS.keys())
    skip = {"__pycache__", "node_modules", ".git", ".venv", "venv",
            "dist", "build", ".next", "target"}
    snippets: list[str] = []

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in skip]
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext in extensions:
                fpath = os.path.join(root, fname)
                try:
                    content = Path(fpath).read_text(encoding="utf-8", errors="replace")[:1500]
                    lang = SUPPORTED_EXTENSIONS[ext]
                    snippets.append(f"### {fpath}\n```{lang}\n{content}\n```")
                except Exception:
                    pass
            if len(snippets) >= 20:
                break
        if len(snippets) >= 20:
            break

    prompt = f"""You are a senior software engineer summarising a codebase.

{chr(10).join(snippets)}

Provide a structured summary:
1. **Architecture** — high-level structure of the system.
2. **Main modules** — each file/module and its role.
3. **Data flow** — how data enters, transforms, and exits.
4. **Key dependencies** — important third-party libraries and what they do.
5. **Potential improvements** — 2–3 concrete, actionable suggestions.

Tone / depth: {mode_instruction}
"""
    return _llm(prompt, max_tokens=2000)