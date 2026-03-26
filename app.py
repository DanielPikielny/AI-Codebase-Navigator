from __future__ import annotations
import os
import re
import shutil
import tempfile
from pathlib import Path
import streamlit as st
from openai import OpenAI
from embeddings import SUPPORTED_EXTENSIONS, create_chunks, embed_chunks, embed_text
from retriever import CodeRetriever
from agent import (
    ask_multistep,
    locate_feature,
    suggest_refactor,
    generate_tests,
    explain_file,
    summarize_repo,
)
from dependency_graph import build_dependency_graph
from memory import ChatMemory

client = OpenAI()

st.set_page_config(
    page_title="Codebase Navigator",
    page_icon="🧭",
    layout="wide",
)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.snippet-card {
    background:#0d1117; border:1px solid #30363d;
    border-radius:10px; margin-bottom:14px; overflow:hidden;
}
.snippet-header {
    background:#161b22; border-bottom:1px solid #30363d;
    padding:8px 14px; display:flex; justify-content:space-between;
    align-items:center; font-size:12px; color:#8b949e; gap:8px;
}
.snippet-header .filename { color:#79c0ff; font-weight:600; }
.snippet-header .lang     { color:#8b949e; font-style:italic; }
.snippet-header .score    { color:#3fb950; font-family:'JetBrains Mono',monospace; white-space:nowrap; }
.signal-pill {
    display:inline-block; padding:2px 7px; border-radius:10px;
    font-size:10px; font-weight:600; margin-left:4px;
}
.signal-semantic { background:#1c2d3f; color:#79c0ff; }
.signal-keyword  { background:#2d1f0e; color:#e3b341; }
.snippet-body {
    padding:14px; font-family:'JetBrains Mono',monospace;
    font-size:13px; line-height:1.6; color:#e6edf3;
    white-space:pre-wrap; word-break:break-word;
    max-height:320px; overflow-y:auto;
}
.snippet-body mark {
    background:rgba(210,153,34,.35); color:#e3b341;
    border-radius:3px; padding:0 2px;
}
.mode-pill {
    display:inline-block; padding:4px 12px; border-radius:20px;
    font-size:12px; font-weight:600; letter-spacing:.03em;
}
.mode-beginner     { background:#1c3a2f; color:#3fb950; border:1px solid #2ea04326; }
.mode-intermediate { background:#1c2d3f; color:#79c0ff; border:1px solid #388bfd26; }
.mode-expert       { background:#2d1f3d; color:#bc8cff; border:1px solid #8957e526; }
.sub-question-card {
    background:#161b22; border-left:3px solid #388bfd;
    border-radius:0 8px 8px 0; padding:12px 16px; margin-bottom:10px;
}
.sub-q-label { font-size:11px; color:#8b949e; margin-bottom:4px; }
.sub-q-text  { font-size:13px; color:#c9d1d9; }
</style>
""", unsafe_allow_html=True)

EXTENSIONS = set(SUPPORTED_EXTENSIONS.keys())
SKIP_DIRS  = {
    "__pycache__", "node_modules", ".git", ".venv", "venv",
    "dist", "build", ".next", "target",
}
MODE_INSTRUCTIONS = {
    "Beginner": (
        "Explain as if to someone new to programming. "
        "Avoid jargon, use analogies and plain English."
    ),
    "Intermediate": (
        "Explain to a developer familiar with Python and common patterns. "
        "Reference standard concepts freely, but explain architecture decisions."
    ),
    "Expert": (
        "Explain to a senior engineer. Be concise and precise. "
        "Focus on non-obvious decisions, trade-offs, and performance. Skip basics."
    ),
}

def _mode_badge(mode: str) -> str:
    css   = {"Beginner": "beginner", "Intermediate": "intermediate", "Expert": "expert"}
    icons = {"Beginner": "🟢", "Intermediate": "🔵", "Expert": "🟣"}
    return f'<span class="mode-pill mode-{css[mode]}">{icons[mode]} {mode}</span>'

def _highlight(code: str, query: str) -> str:
    for term in (t for t in re.split(r"\W+", query) if len(t) > 3):
        code = re.sub(rf"(?i)({re.escape(term)})", r"<mark>\1</mark>", code)
    return code

def _signal_pills(signals: list[str]) -> str:
    html = ""
    for sig in signals:
        html += f'<span class="signal-pill signal-{sig}">{sig}</span>'
    return html

def _render_snippet(result: dict, query: str = "") -> None:
    source   = result["source"]
    text     = result.get("text", "")
    lang     = result.get("language", "")
    sim      = result.get("similarity", 0.0)
    signals  = result.get("match_signals", [])
    filename = os.path.basename(source)
    body     = _highlight(text, query) if query else text

    st.markdown(f"""
<div class="snippet-card">
  <div class="snippet-header">
    <span class="filename"> {filename}</span>
    <span class="lang">{lang}</span>
    <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
                 font-size:11px">{source}</span>
    <span>{_signal_pills(signals)}</span>
    <span class="score">sim {sim:.0%}</span>
  </div>
  <div class="snippet-body">{body}</div>
</div>""", unsafe_allow_html=True)

def _load_files(repo_path: str) -> list[str]:
    files: list[str] = []
    for root, dirs, fnames in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in sorted(fnames):
            if os.path.splitext(f)[1].lower() in EXTENSIONS:
                files.append(os.path.join(root, f))
    return files
def _load_qa_prompt() -> str:
    return (Path("prompts") / "qa_prompt.txt").read_text(encoding="utf-8")

def _is_github_url(text: str) -> bool:
    return bool(re.match(r"https?://github\.com/[\w.-]+/[\w.-]+", text.strip()))

def _clone_repo(url: str, token: str | None = None) -> str:
    try:
        from git import Repo
    except ImportError:
        raise RuntimeError("Run: pip install gitpython")
    clone_url = url.strip().rstrip("/")
    if token:
        clone_url = clone_url.replace("https://", f"https://{token}@")
    dest = tempfile.mkdtemp(prefix="codebase_nav_")
    try:
        Repo.clone_from(clone_url, dest, depth=1)
    except Exception as exc:
        shutil.rmtree(dest, ignore_errors=True)
        raise RuntimeError(f"Git clone failed: {exc}") from exc
    return dest

@st.cache_resource(show_spinner=False)
def _build_index(repo_path: str) -> CodeRetriever:
    ph = st.empty()
    retriever = CodeRetriever.build_or_load(
        repo_path        = repo_path,
        create_chunks_fn = create_chunks,
        embed_chunks_fn  = embed_chunks,
        extensions       = EXTENSIONS,
        progress_fn      = lambda m: ph.info(m),
    )
    ph.empty()
    return retriever

for key, default in {
    "repo_path":    None,
    "clone_dir":    None,
    "source_label": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.header("Settings")

    mode = st.radio("Explanation mode", ["Beginner", "Intermediate", "Expert"], index=1)
    st.markdown(_mode_badge(mode), unsafe_allow_html=True)
    st.divider()

    use_hybrid = st.toggle(
        "Hybrid search (semantic + keyword)",
        value=True,
        help=(
            "Combines FAISS vector search with BM25 keyword search using "
            "Reciprocal Rank Fusion. Better recall for exact identifiers "
            "and class names. Disable to use semantic search only."
        ),
    )

    st.divider()
    st.subheader("Repository")
    src_type   = st.radio("Input type", ["Local path", "GitHub URL"], horizontal=True)
    repo_input = st.text_input(
        "Path or URL",
        placeholder=(
            "/path/to/repo" if src_type == "Local path"
            else "https://github.com/user/repo"
        ),
    )
    gh_token = ""
    if src_type == "GitHub URL":
        gh_token = st.text_input("GitHub token (private repos)", type="password")

    if st.button("Load repository", type="primary", use_container_width=True):
        if repo_input.strip():
            if st.session_state.clone_dir and os.path.isdir(st.session_state.clone_dir):
                shutil.rmtree(st.session_state.clone_dir, ignore_errors=True)
                st.session_state.clone_dir = None

            resolved = repo_input.strip()
            if src_type == "GitHub URL":
                if not _is_github_url(resolved):
                    st.error("Not a valid GitHub URL.")
                    st.stop()
                with st.spinner("Cloning…"):
                    try:
                        resolved = _clone_repo(resolved, gh_token or None)
                        st.session_state.clone_dir = resolved
                    except RuntimeError as exc:
                        st.error(str(exc))
                        st.stop()
            elif not os.path.isdir(resolved):
                st.error("Path not found.")
                st.stop()

            st.session_state.repo_path    = resolved
            st.session_state.source_label = repo_input.strip()
            st.rerun()

    if st.session_state.source_label:
        st.caption(f"Active: `{st.session_state.source_label}`")

    if st.session_state.clone_dir:
        if st.button("🗑 Remove cloned repo", use_container_width=True):
            shutil.rmtree(st.session_state.clone_dir, ignore_errors=True)
            for k in ("clone_dir", "repo_path"):
                st.session_state[k] = None
            st.session_state.source_label = ""
            st.rerun()

st.title("Codebase Navigator")
repo_path = st.session_state.repo_path
if not repo_path:
    st.info("Load a repository from the sidebar to get started.")
    st.stop()

retriever   = _build_index(repo_path)
qa_template = _load_qa_prompt()
mode_instr  = MODE_INSTRUCTIONS[mode]
all_files   = _load_files(repo_path)

def _retrieve(query: str, k: int = 5) -> list[dict]:
    emb = embed_text(query)
    if use_hybrid:
        return retriever.hybrid_search(query, emb, k=k)
    else:
        return retriever.search(emb, k=k)

tab_chat, tab_locate, tab_graph, tab_actions, tab_summary = st.tabs([
    "Chat",
    "Locate feature",
    "Dependency graph",
    "Code actions",
    "Repo summary",
])


with tab_chat:
    memory = ChatMemory.from_session(st.session_state, max_turns=6)
    col_agent, col_clear = st.columns([4, 1])
    with col_agent:
        use_agent = st.toggle(
            "Multi-step reasoning",
            value=True,
            help="Decomposes your question, retrieves evidence per sub-question, "
                 "then synthesises. More thorough; 3–5× more LLM calls.",
        )
    with col_clear:
        if st.button("🗑 Clear chat", use_container_width=True):
            memory.clear()
            memory.to_session(st.session_state)
            st.rerun()
    memory.render_history(st)
    query = st.chat_input("Ask a question about the codebase…")

    if query:
        with st.chat_message("user"):
            st.markdown(query)
        memory_context = memory.as_context()

        with st.chat_message("assistant"):
            status = st.empty()

            if use_agent:
                augmented_template = qa_template
                if memory_context:
                    augmented_template = (
                        qa_template.replace(
                            "Code Context:",
                            f"{memory_context}\n\nCode Context:",
                        )
                    )

                result = ask_multistep(
                    retriever        = retriever,
                    question         = query,
                    mode_instruction = mode_instr,
                    qa_template      = augmented_template,
                    progress_fn      = lambda m: status.info(m),
                    retrieve_fn      = lambda q, k: _retrieve(q, k),
                )
                status.empty()

                st.markdown(result.final_answer)
                answer  = result.final_answer
                sources = result.all_sources

                with st.expander("🔬 Reasoning steps"):
                    for i, sa in enumerate(result.sub_answers, 1):
                        st.markdown(f"""
<div class="sub-question-card">
  <div class="sub-q-label">Sub-question {i}</div>
  <div class="sub-q-text">{sa.question}</div>
</div>""", unsafe_allow_html=True)
                        st.markdown(sa.answer)
                        if i < len(result.sub_answers):
                            st.divider()

            else:
                results = _retrieve(query, k=5)
                context = "\n\n".join(
                    f"{r['source']}:\n{r['text']}" for r in results
                )
                prompt = (
                    (f"{memory_context}\n\n" if memory_context else "")
                    + qa_template.format(context=context, question=query)
                    + f"\n\nTone / depth: {mode_instr}"
                )
                response = client.chat.completions.create(
                    model    = "gpt-4o-mini",
                    messages = [{"role": "user", "content": prompt}],
                )
                answer  = response.choices[0].message.content
                sources = results
                status.empty()
                st.markdown(answer)

            if sources:
                with st.expander(f"Sources ({len(sources)})"):
                    for s in sources:
                        _render_snippet(s, query)
        memory.add(query, answer, sources)
        memory.to_session(st.session_state)


with tab_locate:
    st.subheader("Where is X implemented?")
    st.caption(
        "Broad search (k=8) ranked by likelihood, with per-candidate explanations."
    )
    locate_q = st.text_input(
        "Feature or concept",
        placeholder="authentication, rate limiting, database connection…",
        label_visibility="collapsed",
        key="locate_input",
    )
    if locate_q:
        with st.spinner("Ranking candidates…"):
            loc = locate_feature(
                retriever        = retriever,
                query            = locate_q,
                mode_instruction = mode_instr,
                k                = 8,
                retrieve_fn      = lambda q, k: _retrieve(q, k),
            )

        from agent import Confidence
        if loc.confidence == Confidence.LOW:
            st.warning(
                "**Low confidence** — the search did not find a strong match. "
                "The result below is speculative.\n\n"
                "**Try refining your query** with a more specific function name, "
                "class name, or filename.",
            )
            with st.expander("Show low-confidence result anyway"):
                st.markdown(loc.answer)
        elif loc.confidence == Confidence.MEDIUM:
            st.info("**Medium confidence** — partial match found.")
            st.markdown(loc.answer)
        else:
            st.success("**High confidence** match found.")
            st.markdown(loc.answer)

        with st.expander("All candidate snippets"):
            for s in loc.sources:
                _render_snippet(s, locate_q)

with tab_graph:
    st.subheader("File dependency graph")
    st.caption(
        "Static import analysis. Third-party packages excluded; "
        "only project-local dependencies shown."
    )
    if st.button("Build graph", key="dep_graph_btn"):
        with st.spinner("Analysing imports…"):
            try:
                adjacency, dot = build_dependency_graph(repo_path)
                total_edges = sum(len(v) for v in adjacency.values())
                isolated    = sum(1 for v in adjacency.values() if not v)
                st.caption(
                    f"{len(adjacency)} files · {total_edges} dependencies · "
                    f"{isolated} isolated"
                )
                st.graphviz_chart(dot.source, use_container_width=True)
                with st.expander("Adjacency list"):
                    for src, deps in sorted(adjacency.items()):
                        if deps:
                            st.markdown(
                                f"**`{src}`** → "
                                + ", ".join(f"`{d}`" for d in sorted(deps))
                            )
            except ImportError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Graph build failed: {exc}")

with tab_actions:
    st.subheader("Code actions")
    if not all_files:
        st.warning("No supported source files found.")
    else:
        selected = st.selectbox(
            "Select a file",
            options     = all_files,
            format_func = lambda p: os.path.relpath(p, repo_path),
            key         = "actions_file",
        )
        col_exp, col_ref, col_tst = st.columns(3)

        if col_exp.button("Explain file", use_container_width=True):
            with st.spinner("Explaining…"):
                st.markdown(explain_file(selected, mode_instr))

        if col_ref.button("Suggest refactoring", use_container_width=True):
            with st.spinner("Analysing…"):
                st.markdown(suggest_refactor(selected, mode_instr))

        if col_tst.button("Generate tests", use_container_width=True):
            with st.spinner("Writing tests…"):
                test_code = generate_tests(selected, mode_instr)
            code_match = re.search(r"```(?:\w+)?\n([\s\S]+?)```", test_code)
            clean      = code_match.group(1) if code_match else test_code
            st.code(clean, language="python")
            st.download_button(
                label     = f"⬇Download test_{os.path.basename(selected)}",
                data      = clean,
                file_name = f"test_{os.path.basename(selected)}",
                mime      = "text/plain",
            )

with tab_summary:
    st.subheader("Repository summary")
    if st.button("Generate summary", key="summary_btn"):
        with st.spinner("Analysing codebase…"):
            st.markdown(summarize_repo(repo_path, mode_instr))