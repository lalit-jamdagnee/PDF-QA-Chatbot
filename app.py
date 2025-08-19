# app.py
import streamlit as st
import fitz  # PyMuPDF
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# -------- PDF Loader -------- #
def extract_text_with_metadata(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({"page": i + 1, "content": text})
    return pages

# -------- FAISS VectorStore -------- #
def build_faiss(pages):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []
    for p in pages:
        chunks = splitter.split_text(p["content"])
        for c in chunks:
            docs.append(
                {"page_content": c, "metadata": {"page": p["page"]}}
            )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # change to "cuda" if GPU available
    )
    vectorstore = FAISS.from_texts(
        [d["page_content"] for d in docs],
        embeddings,
        metadatas=[d["metadata"] for d in docs]
    )
    return vectorstore, docs

# -------- Summarizer -------- #
def summarize_document(docs, llm, max_chunks=100):
    text_sample = " ".join([d["page_content"] for d in docs[:max_chunks]])
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following document in a structured, hierarchical way:\n\n{text}"
    )
    summary = llm.invoke(prompt.format_messages(text=text_sample))
    return summary.content

# -------- Streamlit UI -------- #
st.set_page_config(page_title="PDF Summarizer & QA", layout="wide")
st.title("📄 Intelligent PDF Summarizer & QA Bot")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    if st.button("📥 Process PDF"):
        with st.spinner("Processing PDF..."):
            pages = extract_text_with_metadata(uploaded_file)
            vectorstore, docs = build_faiss(pages)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            llm = ChatNVIDIA(model="meta/llama3-8b-instruct")

            st.session_state.retriever = retriever
            st.session_state.llm = llm
            st.session_state.summary = summarize_document(docs, llm)
            st.session_state.messages = []

# --- If PDF is processed --- #
if "summary" in st.session_state:
    st.subheader("📌 Document Summary")
    st.write(st.session_state.summary)

    st.subheader("💬 Chat with your PDF")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask a question about the document...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        docs = st.session_state.retriever.get_relevant_documents(query)
        context = "\n\n".join(
            [f"(Page {d.metadata['page']}): {d.page_content}" for d in docs]
        )

        prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant. Use the following context to answer the question.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer in detail with page citations:"
        )
        response = st.session_state.llm.invoke(
            prompt.format_messages(context=context, question=query)
        )

        answer = response.content
        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.markdown(answer)
