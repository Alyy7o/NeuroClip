import json
from pathlib import Path
import sys

def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def embed_json(json_path_str: str):
    p = Path(json_path_str)
    data = json.loads(p.read_text(encoding="utf-8"))
    sentences = data.get("sentences") or []
    texts = [s.get("sentence", "") for s in sentences]
    if not texts:
        return None
    vecs = get_model().encode(texts)
    emb = {"vectors": [list(map(float, v)) for v in vecs], "sentences": sentences}
    out = p.with_name(p.stem.replace(".v4", "") + ".embeddings.json")
    out.write_text(json.dumps(emb, ensure_ascii=False), encoding="utf-8")
    return str(out)

def main():
    if len(sys.argv) < 2:
        sys.exit(1)
    out = embed_json(sys.argv[1])
    if out:
        print(out)

if __name__ == "__main__":
    main()