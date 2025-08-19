# 📄 Intelligent PDF Summarizer & QA Bot
See live demo: https://huggingface.co/spaces/lalitJamdagnee/PDF_QA_Chatbot
A Streamlit-based app that lets you:

- Upload a **PDF document**
- Get an **automatic summary** (chunked + hierarchical to handle long docs)
- **Chat with your PDF** using semantic search + an LLM retriever

Built with:
- [Streamlit](https://streamlit.io/) for the UI  
- [LangChain](https://www.langchain.com/) for text splitting, embeddings & prompts  
- [FAISS](https://github.com/facebookresearch/faiss) for vector search  
- [NVIDIA LLM API](https://build.nvidia.com/nvidia) as the chat model backend  
- [HuggingFace Sentence Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) for embeddings  

---

## 🚀 Features
1. **PDF Processing**  
   - Extracts text page by page with [PyMuPDF](https://pymupdf.readthedocs.io/)  
   - Splits into overlapping chunks for better retrieval  

2. **Summarization**  
   - Each chunk is summarized separately  
   - Summaries are combined into a final structured summary  

3. **Question Answering**  
   - User queries are embedded and matched with relevant PDF chunks via FAISS  
   - Retrieved context is sent to the LLM to generate detailed answers with **page citations**  

---

