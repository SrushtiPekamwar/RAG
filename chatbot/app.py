import re
import streamlit as st
from rag_core import RAG, TOP_N, GOLD, highlight_html

st.set_page_config(page_title="D&D Rules Oracle: Baseline vs Re-Ranking", page_icon="🐉", layout="wide")

# ---- D&D theme: parchment background, dark-red and gold accents, fantasy heading font ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel+Decorative:wght@700&family=Cinzel:wght@500;600;700&family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400&display=swap');

/* Body text: EB Garamond (readable old-style serif). Applied to text elements only, so Streamlit's icon fonts keep working. */
html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button p,
.stApp summary p, .stApp td, .stApp th, [data-testid="stCaptionContainer"], [data-testid="stMarkdownContainer"] {
    font-family: 'EB Garamond', Georgia, 'Times New Roman', serif;
}
.stApp p, .stApp li, .stApp td, .stApp th, .stApp input, .stApp textarea { font-size: 1.12rem; line-height: 1.5; }
[data-testid="stCaptionContainer"] p { font-size: 1rem; font-style: italic; }

/* Headings: Cinzel (classic fantasy capitals) */
h1, h2, h3, h4, .panel-head, .dont-know, .stApp label p {
    font-family: 'Cinzel', Georgia, serif !important; letter-spacing: .4px;
}
h1, h2, h3, h4 { color: #5a0f0f; font-weight: 700; }
.stApp label p { font-size: 1.05rem; font-weight: 600; color: #5a0f0f; }

.stApp { background: radial-gradient(circle at top, #f8efd9 0%, #efe0bd 100%); }
[data-testid="stSidebar"] { background: #2b1d0e; border-right: 4px double #c9a227; }
[data-testid="stSidebar"] * { color: #f3e6c4 !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #ffd86b !important; }
[data-testid="stSidebar"] .stButton button { background: #3d2a14; border: 1px solid #c9a227; text-align: left; }
[data-testid="stSidebar"] .stButton button p { font-size: 1.02rem; line-height: 1.3; }
[data-testid="stSidebar"] .stButton button:hover { background: #5a0f0f; border-color: #ffd86b; }

.oracle-banner { border: 3px double #c9a227; border-radius: 10px; padding: 16px 22px; margin-bottom: 14px;
                 background: linear-gradient(135deg, #5a0f0f 0%, #3a0a0a 100%); color: #f7e7b6; }
.oracle-banner .oracle-title { font-family: 'Cinzel Decorative', 'Cinzel', Georgia, serif !important; color: #ffd86b !important;
                               font-size: 2.2rem; font-weight: 700; letter-spacing: 1px; margin: 0; }
.oracle-banner p { margin: 6px 0 0 0; color: #f3e6c4; font-size: 1.15rem; }

.panel-head { border-left: 8px solid var(--c); padding: 6px 12px; margin: 4px 0 8px 0; border-radius: 4px;
              background: rgba(255,255,255,.45); font-weight: 700; color: var(--c); font-size: 1.15rem; }
.dont-know { border: 2px dashed #6b4fa0; background: #efe6fb; color: #3b2370; border-radius: 8px; padding: 8px 12px; margin-bottom: 6px;
             font-size: 1rem; font-weight: 600; }

[data-testid="stExpander"] { background: rgba(255,255,255,.55); border: 1px solid #c9a227; border-radius: 6px; }
[data-testid="stExpander"] summary p { font-family: 'Cinzel', Georgia, serif; font-size: .95rem; font-weight: 600; }
[data-testid="stVerticalBlockBorderWrapper"] { background: rgba(255,250,235,.8); }
[data-testid="stTable"] th { font-family: 'Cinzel', Georgia, serif; }
mark { font-family: inherit; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="🎲 Rolling initiative: loading models and indexing the rules...")
def get_rag():
    return RAG()


rag = get_rag()

# Example questions, tagged by the failure mode each one tests (must match the keys of GOLD)
EXAMPLES = [
    ("🗡️", "Does casting a spell as a ritual use up a spell slot?"),
    ("🗡️", "Does moving via magical transportation trigger an opportunity attack?"),
    ("🗡️", "What happens if you have both advantage and disadvantage on the same roll?"),
    ("🛡️", "Can healing give me more hit points than my maximum?"),
    ("🛡️", "Does a stable creature at 0 hit points still have to make death saving throws?"),
    ("🛡️", "What is the limitation on how often you can use the Cleave property in a single turn?"),
    ("🗝️", "How much does moving through rough ground increase the cost of each foot traveled?"),
    ("📜", "What specific action does the Game Master usually request when dice are involved in a social exchange?"),
    ("📜", "How much does half cover increase my armor class?"),
    ("⚠️", "What penalty do you suffer on ability checks if you wear armor you are not proficient with?"),
    ("⚠️", "How much damage does a makeshift object deal when it hits a target?"),
    ("🎲", "What happens when I roll a natural 20 on a death saving throw?"),
    ("🎲", "Can I take another Reaction if I already used my Reaction this round?"),
    ("🧙", "How much does the D&D Player’s Handbook cost in dollars?"),
]
assert {q for _, q in EXAMPLES} == set(GOLD), "EXAMPLES and GOLD are out of sync"

DONT_KNOW = re.compile(r"(do(es)? not|don.t|doesn.t) contain|not (mentioned|stated|found|covered)|no information|i (do not|don.t) know", re.I)


def says_unknown(text):
    return bool(DONT_KNOW.search(text))


st.markdown("""
<div class="oracle-banner">
  <div class="oracle-title">🐉 D&amp;D 5e Rules Oracle</div>
  <p>Baseline retrieval versus cross-encoder re-ranking on the System Reference Document 5.2.1
  (Playing the Game, Equipment, Rules Glossary). Both pipelines search the same chunks and use the same prompt.
  Only the 3 chunks given to the model differ.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🎲 Choose your quest")
    st.caption("Each question tests one failure mode")
    for i, (icon, ex) in enumerate(EXAMPLES):
        if st.button(f"{icon} {ex}", key=f"ex{i}", use_container_width=True):
            st.session_state["q"] = ex
    st.markdown("---")
    st.markdown(
        "**Legend**\n\n"
        "🗡️ exception / corner case\n\n🛡️ negation / constraint\n\n🗝️ keyword trap\n\n"
        "📜 buried chunk\n\n⚠️ re-ranking hurts\n\n🎲 easy control\n\n🧙 not in the document\n\n"
        "---\n\n"
        "🟠 chunk is in the top 3 sent to the LLM\n\n"
        "🎯 chunk contains the answer phrase (test questions only)\n\n"
        "Yellow = the answer phrase. Light blue = your question's keywords (free-typed questions)."
    )
    st.markdown("---")
    st.caption("Contains material from the SRD 5.2.1 by Wizards of the Coast LLC, licensed under CC BY 4.0.")

query = st.text_input("📖 Ask a question about the D&D 5e rules", key="q")

if query:
    res = rag.retrieve_both(query)
    base, rer = res["base"], res["reranked"]

    with st.spinner("🔮 Consulting the oracle (generating both answers)..."):
        ans1 = rag.generate(query, base[:TOP_N])
        ans2 = rag.generate(query, rer[:TOP_N])

    c1, c2 = st.columns(2)

    def show(col, title, color, ids, scores, score_name, other, answer, ms, is_rerank):
        with col:
            st.markdown(f'<div class="panel-head" style="--c:{color}">{title}</div>', unsafe_allow_html=True)
            if says_unknown(answer):
                st.markdown('<div class="dont-know">🧙 The oracle admits it does not know: '
                            'the rules it was shown do not contain this answer.</div>', unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(answer)
            st.caption(f"Retrieval time: {ms:.1f} ms")
            st.markdown("**Top 10 ranking** (top 3 go to the LLM)")
            for rank, (cid, sc) in enumerate(zip(ids, scores), start=1):
                other_rank = other.index(cid) + 1
                moved = other_rank - rank  # positive: this pipeline ranks it higher
                tag = ""
                if is_rerank and moved != 0:
                    tag = f" | was #{other_rank} in baseline " + ("⬆️" if moved > 0 else "⬇️")
                body, has_gold = highlight_html(rag.chunks[cid]["text"], query)
                label = f"#{rank} chunk {cid} (page {rag.chunks[cid]['page']}) | {score_name} {sc:.3f}{tag}"
                if has_gold:
                    label = "🎯 " + label
                with st.expander(("🟠 " if rank <= TOP_N else "") + label, expanded=False):
                    st.markdown(body, unsafe_allow_html=True)

    show(c1, "🎲 Pipeline 1: Baseline (bi-encoder)", "#1f4e79", base, res["base_scores"], "cosine", rer, ans1,
         res["t_base_ms"], False)
    show(c2, "⚔️ Pipeline 2: Re-ranked (cross-encoder)", "#1b5e20", rer, res["ce_scores"], "CE score", base, ans2,
         res["t_base_ms"] + res["t_ce_ms"], True)

    st.divider()
    st.markdown("### ⏳ Latency")
    st.table({"Stage": ["Baseline retrieval", "Cross-encoder (10 pairs)", "Re-ranking total"],
              "ms": [f"{res['t_base_ms']:.1f}", f"{res['t_ce_ms']:.1f}",
                     f"{res['t_base_ms'] + res['t_ce_ms']:.1f}"]})
else:
    st.info("📜 Type a question, or pick a quest from the sidebar.")
