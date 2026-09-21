# ⚖️ Ethiopian Legal RAG Assistant

A semantic search and RAG (Retrieval-Augmented Generation) application for Ethiopian legal documents.

## Overview

This platform allows users to search and query Ethiopian law using natural language. It indexes major legal codes and the Constitution, then uses AI to provide accurate, cited answers to legal questions.

## Features

- **Full Legal Corpus**: Civil Code, Criminal Code, Commercial Code, Family Code, Maritime Code, Civil Procedure Code, and Constitution
- **Hierarchical Retrieval**: Two-step search for more accurate results
- **Instant AI Answers**: Natural language questions get cited responses
- **User Authentication**: Secure accounts with conversation history
- **Theme Toggle**: Dark and light mode support

## Tech Stack

- **Frontend**: Streamlit
- **Database**: PostgreSQL with pgvector
- **Embeddings**: Sentence Transformers (MiniLM-L6-v2)
- **LLM**: OpenRouter API
- **PDF Processing**: PyMuPDF for legal document parsing

## Setup

### Prerequisites

- Python 3.10+
- PostgreSQL with pgvector extension
- OpenRouter API key

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd first-rag
```

2. Create virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file with your credentials:
```env
DATABASE_URL=postgresql://user:password@localhost:5432/legal_db
OPENROUTER_API_KEY=your_key_here
```

5. Initialize the database:
```bash
python db_pgvector.py
```

6. Ingest legal documents:
```bash
python ingest_to_pg.py
```

7. Run the application:
```bash
streamlit run main.py
```

## Project Structure

```
first-rag/
├── main.py                  # Streamlit UI
├── db_pgvector.py          # Database operations
├── auth.py                 # User authentication
├── embedding.py            # Embedding generation
├── local_llm.py            # LLM interface
├── legal_pdf_parser.py     # PDF processing
├── ingest_to_pg.py         # Data ingestion script
├── pdfs/                   # Legal documents
│   ├── Ethiopian-Codes/
│   └── metadata.json
├── assets/                 # UI assets
│   ├── logo.png
│   └── demo.mp4
└── requirements.txt
```

## Usage

1. **Sign Up**: Create an account on the landing page
2. **Ask Questions**: Type legal questions in plain language
3. **View Citations**: Get answers with article references
4. **Browse History**: Access previous conversations from sidebar
5. **Toggle Theme**: Switch between dark/light mode

## Evaluation

The system includes evaluation tools:

- `eval_retrieval.py`: Test retrieval accuracy
- `generate_questions.py`: Create test datasets
- `compare_pg.py`: Compare retrieval strategies

## Development

### Adding New Legal Documents

1. Place PDFs in `pdfs/` directory
2. Update `pdfs/metadata.json` with document info
3. Run ingestion: `python ingest_to_pg.py`

### Configuration

Edit these parameters in `main.py`:
- `TOP_CONTAINERS`: Number of containers to retrieve (default: 3)
- `CHUNKS_PER_CONT`: Chunks per container (default: 9)
- `MAX_CHUNKS`: Maximum total chunks (default: 12)

## License

© 2025 RARAS Technology

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Support

For issues or questions, please contact the development team.
