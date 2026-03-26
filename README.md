# Codebase Navigator

Codebase Navigator is a high-precision RAG (Retrieval-Augmented Generation) engine designed specifically for navigating, understanding, and refactoring complex source code.

While standard RAG systems treat code like generic text, Codebase Navigator utilizes Hybrid Retrieval, Multi-step Agentic Reasoning, and Static Analysis to provide architecturally aware answers. It is built to think like a Senior Engineer—breaking down complex queries into logical research steps before synthesizing a final response.

Key Features
1. Hybrid Search & Fusion

Codebases rely on both conceptual intent and literal identifiers. The system implements a dual-retrieval strategy:

    Semantic Search: Uses OpenAI’s text-embedding-3-small to find chunks based on meaning.

    Keyword Precision: Employs Rank-BM25 to catch specific function names, variables, and identifiers that semantic embeddings might overlook.

    Reciprocal Rank Fusion (RRF): Mathematically merges the results from both methods to ensure the most relevant code is prioritized.

2. Multi-Step Agent Reasoning

The core agent doesn't just "guess." It follows a structured engineering workflow:

    Decomposition: Breaks high-level questions into 2–4 independently searchable sub-questions.

    Feature Localization: Ranks candidate snippets and identifies the "ground truth" implementation files.

    Synthesis: Integrates research findings into a coherent technical explanation, explicitly identifying design patterns (e.g., Singleton, Factory, Strategy) and data flows.

3. Static Analysis & Dependency Mapping

The system uses Python’s ast (Abstract Syntax Tree) module to parse the codebase and build a real-time adjacency list of imports. This is visualized via Graphviz, allowing developers to see a structural map of how files interact.
4. Developer Workflows

    Refactoring Suite: Provides suggestions prioritized by impact: Correctness > Performance > Readability.

    Automated Testing: Generates pytest files with mandatory mocking for external dependencies (APIs, DBs, I/O).

    Repo Summarization: Quickly generates an architectural overview of a new repository.

Architecture
Plaintext

├── app.py                # Streamlit UI & Orchestration

├── agent.py              # Multi-step reasoning logic & LLM interaction

├── retriever.py          # Hybrid Search (FAISS + BM25) & RRF Fusion

├── embeddings.py         # Async embedding pipeline & file chunking

├── dependency_graph.py   # AST-based static analysis & Graphviz rendering

├── memory.py             # Sliding-window conversation state

└── prompts/              # Specialized "Senior Engineer" prompt engineering

Tech Stack

    Orchestration: Python, Streamlit

    LLM: OpenAI GPT-4o-mini

    Vector Store: FAISS

    Search: Rank-BM25

    Static Analysis: Python ast + Graphviz

    Async I/O: asyncio for high-throughput embedding generation

Getting Started
Prerequisites

    Python 3.9+

    OpenAI API Key

    Graphviz (installed on your system for dependency visualization)

Installation

    Clone the repository:
    Bash

    git clone https://github.com/yourusername/codebase-navigator.git
    cd codebase-navigator

    Install dependencies:
    Bash

    pip install -r requirements.txt

    Set your OpenAI API Key:
    Bash

    export OPENAI_API_KEY='your-key-here'

Running the App
Bash

streamlit run app.py
