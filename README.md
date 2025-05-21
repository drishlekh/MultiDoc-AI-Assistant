# MultiDoc-AI-Assistant

MultiDoc-AI-Assistant is an intelligent application that allows you to build a knowledge base from various document types (PDFs, CSVs, JSON, websites, handwritten notes/images) and then chat with this knowledge base using a conversational AI. It leverages Retrieval Augmented Generation (RAG) to provide contextually relevant answers based on your uploaded sources.

## Features ✨

*   **Multi-Source Ingestion**: Upload and process various file types:
    *   PDFs (text-based and scanned/image-based via OCR)
    *   CSV files
    *   JSON files
    *   Images (PNG, JPG, JPEG - for OCR)
    *   Handwritten notes (PDFs or images - for OCR)
    *   Website URLs
*   **Intelligent Chat Interface**: Ask questions and receive answers grounded in the content of your uploaded documents.
*   **Source Referencing**: Assistant's responses can include references to the source documents used to generate the answer.
*   **Dynamic Knowledge Base**: Each processing action creates a fresh, isolated knowledge base for your chat session.
*   **Rate Limiting**: Basic rate limiting for OCR operations to manage resource usage.

## Tech Stack 🛠️

This project utilizes a modern stack for document processing, AI, and web application development:

*   **Backend & Application Logic**:
    *   Python
    *   Streamlit: For the interactive web application interface.
*   **Large Language Model (LLM) & Orchestration**:
    *   Langchain: Framework for developing applications powered by language models.
        *   ConversationalRetrievalChain: For implementing the RAG pattern.
        *   ConversationBufferMemory: To maintain chat history.
    *   Groq API (Llama 3): For fast LLM inference.
*   **Document Processing & Text Extraction**:
    *   PyPDF2: For extracting text from text-based PDFs.
    *   `pdf2image` & Poppler: For converting PDF pages to images.
    *   Google Cloud Vision API: For Optical Character Recognition (OCR) on images and scanned PDFs.
    *   Beautiful Soup: For parsing and extracting text from website URLs.
    *   Pandas: For handling CSV data.
*   **Vector Store & Embeddings (RAG Core)**:
    *   ChromaDB: As the vector database to store document embeddings.
    *   Sentence Transformers (`BAAI/bge-small-en-v1.5`): For generating text embeddings.
    *   `pysqlite3-binary`: To ensure SQLite compatibility for ChromaDB in cloud environments.
*   **Deployment**:
    *   Streamlit Cloud 
