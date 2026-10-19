# D&D 5e Rules RAG: Baseline vs Cross-Encoder Re-Ranking

Compares two retrieval pipelines over the *Dungeons & Dragons* 5e System Reference Document 5.2.1 (`data/dnd_srd.pdf`). Three rules sections are indexed (Playing the Game pp. 5-18, Equipment pp. 89-103, Rules Glossary pp. 176-191), which gives 188 chunks.
1. **Baseline**: bi-encoder (all-MiniLM-L6-v2), top 10, top 3 go to the LLM.
2. **Re-ranked**: the same top 10 re-scored by a cross-encoder (ms-marco-MiniLM-L-6-v2), top 3 go to the LLM.

## Setup
```bash
pip install pypdf sentence-transformers requests pandas numpy jupyter streamlit
echo "GROQ_API_KEY=your_key_here" > .env   # free key from console.groq.com; .env is git-ignored
```

## Run the notebook
```bash
jupyter notebook DnD_RAG.ipynb
```

## Run the chatbot UI
```bash
cd chatbot
streamlit run app.py
```

## Files
- `DnD_RAG.ipynb`: all code, 14 test queries, rank tables, answers, latency
- `chatbot/`: Streamlit side-by-side demo
- `Report.pdf`: written report

## Attribution
This work includes material from the System Reference Document 5.2.1 ("SRD 5.2.1") by Wizards of the Coast LLC, available at https://www.dndbeyond.com/srd. The SRD 5.2.1 is licensed under the Creative Commons Attribution 4.0 International License, available at https://creativecommons.org/licenses/by/4.0/legalcode.
