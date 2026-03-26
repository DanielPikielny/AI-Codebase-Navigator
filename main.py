from embeddings import create_chunks, embed_chunks, embed_text
from retriever import CodeRetriever
from openai import OpenAI
import hashlib, json, os

client = OpenAI()
INDEX_CACHE = ".index_cache"


def load_prompt():
    with open("prompts/qa_prompt.txt", "r") as f:
        return f.read()


def _repo_hash(repo_path: str) -> str:
    """Cheap fingerprint: mtime + size of every .py file."""
    sig = []
    for root, _, files in os.walk(repo_path):
        for f in sorted(files):
            if f.endswith(".py"):
                p = os.path.join(root, f)
                s = os.stat(p)
                sig.append(f"{p}:{s.st_mtime}:{s.st_size}")
    return hashlib.md5(json.dumps(sig).encode()).hexdigest()


def build_index(repo_path: str) -> CodeRetriever:
    cache_key = _repo_hash(repo_path)
    cache_dir = os.path.join(INDEX_CACHE, cache_key)

    if os.path.exists(cache_dir):
        print("Loading cached index...")
        return CodeRetriever.load(cache_dir)

    print("Loading and chunking code...")
    chunks = create_chunks(repo_path)

    print(f"Embedding {len(chunks)} chunks...")
    embeddings = embed_chunks(chunks)  

    retriever = CodeRetriever()
    retriever.add(embeddings, chunks)
    retriever.save(cache_dir)          
    return retriever


def ask_question(retriever: CodeRetriever, query: str, prompt_template: str):
    query_embedding = embed_text(query)
    results = retriever.search(query_embedding)

    context = "\n\n".join(
        [f"{r['source']}:\n{r['text']}" for r in results]
    )

    prompt = prompt_template.format(context=context, question=query)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content, results


if __name__ == "__main__":
    repo_path = input("Enter path to repo: ")
    retriever = build_index(repo_path)
    prompt_template = load_prompt()

    while True:
        query = input("\nAsk a question (or 'exit'): ")
        if query.lower() == "exit":
            break

        answer, sources = ask_question(retriever, query, prompt_template)
        print("\nAnswer:\n", answer)

        print("\nSources:")
        for s in sources:
            print(f"  [{s['score']:.2f}] {s['source']}")