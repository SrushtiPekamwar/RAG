"""Same chunking, retrieval, re-ranking and prompt as ../DnD_RAG.ipynb, packaged for the UI."""
import os, re, time
import numpy as np
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
PDF_PATH   = os.path.join(HERE, "..", "data", "dnd_srd.pdf")
ENV_PATH   = os.path.join(HERE, "..", ".env")
# Sections of the SRD that are indexed (PDF page numbers, inclusive)
SECTIONS = [(5, 18, "Playing the Game"), (89, 103, "Equipment"), (176, 191, "Rules Glossary")]
CHUNK_SIZE, OVERLAP = 200, 40
BI_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
CE_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL  = "qwen/qwen3.8-27b"
TOP_K, TOP_N = 10, 3
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

PROMPT = """You are answering a question about the Dungeons & Dragons 5e rules using ONLY the rulebook excerpts below.
If the excerpts do not contain the answer, say so instead of guessing.

Rulebook excerpts:
{context}

Question: {question}
Answer concisely."""


# Known test questions -> the rulebook phrase(s) that answer them (same ones as the notebook).
# Used by the UI to highlight why a chunk matters.
GOLD = {
    "Does casting a spell as a ritual use up a spell slot?": ["doesn’t expend a spell slot"],
    "Does moving via magical transportation trigger an opportunity attack?":
        ["don’t provoke an Opportunity Attack when you Teleport", "teleportation never provokes Opportunity Attacks"],
    "What happens if you have both advantage and disadvantage on the same roll?": ["Advantage and Disadvantage on the same roll cancel each other"],
    "Can healing give me more hit points than my maximum?": ["Your Hit Points can’t exceed your Hit Point maximum"],
    "Does a stable creature at 0 hit points still have to make death saving throws?": ["A Stable creature doesn’t make Death Saving Throws"],
    "What is the limitation on how often you can use the Cleave property in a single turn?": ["You can make this extra attack only once per turn"],
    "How much does moving through rough ground increase the cost of each foot traveled?": ["every foot of movement in that space costs 1 extra foot"],
    "What specific action does the Game Master usually request when dice are involved in a social exchange?": ["ask you to take the Influence action"],
    "How much does half cover increase my armor class?": ["+2 bonus to AC and Dexterity saving throws"],
    "What penalty do you suffer on ability checks if you wear armor you are not proficient with?": ["you have Disadvantage on any D20 Test"],
    "How much damage does a makeshift object deal when it hits a target?": ["the weapon deals 1d4 damage"],
    "What happens when I roll a natural 20 on a death saving throw?": ["If you roll a 20 on the d20, you regain 1 Hit Point"],
    "Can I take another Reaction if I already used my Reaction this round?": ["you can’t take another one until the start of your next turn"],
    # Not in the document: no answer phrase exists, so nothing gets a 🎯; the LLM should decline to answer.
    "How much does the D&D Player’s Handbook cost in dollars?": [],
}

STOPWORDS = set("a an the i my me of to in on is are do does did can be it its if or and for with when what how who which "
                "that this there at by as from have has had not am was were will would should could than then so".split())


def highlight_html(text, query):
    """HTML for a chunk: gold phrase in yellow (if the query is a known test query),
    otherwise the query's keywords in light blue."""
    import html
    gold = GOLD.get(query.strip())
    spans = []
    if gold:
        spans = sorted((m.start(), m.end(), "gold") for g in gold for m in re.finditer(re.escape(g), text, flags=re.I))
    if not spans:
        words = [w for w in re.findall(r"[a-z0-9']+", query.lower()) if w not in STOPWORDS and len(w) > 2]
        if words:
            pat = r"\b(" + "|".join(re.escape(w) for w in sorted(set(words), key=len, reverse=True)) + r")\w*"
            spans = [(m.start(), m.end(), "kw") for m in re.finditer(pat, text, flags=re.I)]
    out, pos = [], 0
    for s, e, kind in spans:
        out.append(html.escape(text[pos:s]))
        style = "background:#fde047;color:#000;font-weight:600" if kind == "gold" else "background:#dbeafe;color:#000"
        out.append(f'<mark style="{style};padding:0 2px;border-radius:3px">{html.escape(text[s:e])}</mark>')
        pos = e
    out.append(html.escape(text[pos:]))
    return "".join(out), bool(gold and spans and spans[0][2] == "gold")


def clean(text):
    text = re.sub(r"System Reference Document 5\.2\.1\s*\d*", " ", text)   # running page header
    text = re.sub(r"(\w)\s?-\n(\w)", r"\1\2", text)                       # re-join words broken across lines
    return re.sub(r"\s+", " ", text).strip()


def build_chunks(path=PDF_PATH, sections=SECTIONS):
    reader, chunks = PdfReader(path), []
    for first, last, name in sections:
        words = []
        for page_no in range(first, last + 1):
            words += [(w, page_no) for w in clean(reader.pages[page_no - 1].extract_text() or "").split()]
        i = 0
        while i < len(words):
            seg = words[i:i + CHUNK_SIZE]
            chunks.append({"id": len(chunks), "section": name, "page": seg[0][1], "text": " ".join(w for w, _ in seg)})
            if i + CHUNK_SIZE >= len(words):
                break
            i += CHUNK_SIZE - OVERLAP
    return chunks


def load_api_key():
    if os.path.exists(ENV_PATH):
        for line in open(ENV_PATH):
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return os.environ.get("GROQ_API_KEY")


class RAG:
    def __init__(self):
        self.chunks = build_chunks()
        self.bi = SentenceTransformer(BI_MODEL)
        self.ce = CrossEncoder(CE_MODEL)
        self.emb = self.bi.encode([c["text"] for c in self.chunks], normalize_embeddings=True)
        self.api_key = load_api_key()
        self.retrieve_both("warm up")

    def retrieve_both(self, query):
        """Returns baseline top-10 and re-ranked top-10 (ids, scores) plus timings in ms."""
        t0 = time.perf_counter()
        scores = self.emb @ self.bi.encode(query, normalize_embeddings=True)
        base = [int(i) for i in np.argsort(-scores)[:TOP_K]]
        t_base = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        ce = self.ce.predict([(query, self.chunks[i]["text"]) for i in base])
        order = np.argsort(-ce)
        reranked = [base[j] for j in order]
        t_ce = (time.perf_counter() - t1) * 1000
        return {"base": base, "base_scores": [float(scores[i]) for i in base],
                "reranked": reranked, "ce_scores": [float(ce[j]) for j in order],
                "t_base_ms": t_base, "t_ce_ms": t_ce}

    def generate(self, question, chunk_ids, retries=3):
        if not self.api_key:
            return "(no GROQ_API_KEY found in ../.env, so generation is skipped)"
        context = "\n\n".join(f"[Excerpt {n+1}, page {self.chunks[i]['page']}] {self.chunks[i]['text']}"
                              for n, i in enumerate(chunk_ids))
        body = {"model": LLM_MODEL, "temperature": 0, "max_tokens": 400,
                "messages": [{"role": "user", "content": PROMPT.format(context=context, question=question)}]}
        for attempt in range(retries):
            try:
                r = requests.post(GROQ_URL, headers={"Authorization": f"Bearer {self.api_key}"}, json=body, timeout=90)
                if r.status_code == 429:
                    time.sleep(float(r.headers.get("retry-after", 20)))
                    continue
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"] or ""
                return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
            except Exception as e:
                last = f"{type(e).__name__}: {str(e)[:120]}"
                time.sleep(5 * (attempt + 1))
        return f"(generation failed: {last})"
