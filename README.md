# DocuNova

**New way to explore documents**

A simple AI assistant that answers natural language questions based on your document collection. Powered by LangChain, OpenAI, ChromaDB, and Gradio.

## Features
- Loads documents from a folder
- Creates embeddings and stores them in a vector database
- Provides a chat interface for question answering

## Quick Start
1. Place your documents in a folder named `documents` inside the project directory.
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the assistant:
   ```sh
   python app.py
   ```

## Requirements
- Python 3.8+
- OpenAI API key (set as environment variable `OPENAI_API_KEY` or in a `.env` file)
