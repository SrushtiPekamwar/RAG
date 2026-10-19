## Setup
```bash
pip install pypdf sentence-transformers requests pandas numpy jupyter streamlit
echo "GROQ_API_KEY=your_key_here" > .env  
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
- `2025201066.pdf`: written report
