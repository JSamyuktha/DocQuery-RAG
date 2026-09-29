import streamlit as st

from rag import (
    extract_text_from_pdf,
    create_chunks,
    create_embeddings,
    store_in_chromadb,
    retrieve_relevant_chunks,
    generate_answer
)

st.set_page_config(
    page_title="DocQuery-RAG",
    page_icon="📄",
    layout="centered"
)

st.title("📄 DocQuery-RAG")
st.write("Upload a PDF and ask questions about its content.")

st.divider()
st.subheader("1. Upload Document")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("Process Document"):

        with st.spinner("Processing document..."):

            pages = extract_text_from_pdf(uploaded_file)

            chunks = create_chunks(pages)

            embeddings = create_embeddings(chunks)

            collection = store_in_chromadb(
                chunks,
                embeddings
            )

            st.session_state.collection = collection
            st.session_state.document_processed = True

        st.success(
            f"Document processed successfully! "
            f"{len(pages)} pages and {len(chunks)} chunks created."
        )

# -------------------------------
#2. Question Answering Section
# -------------------------------

if st.session_state.get("document_processed", False):

    st.divider()
    st.subheader("2. Ask Questions")

    question = st.text_input(
        "Enter your question about the document"
    )

    if st.button("Ask Question"):

        if question.strip():

            with st.spinner("Searching document and generating answer..."):

                retrieved_chunks = retrieve_relevant_chunks(
                    question,
                    st.session_state.collection
                )

                answer = generate_answer(
                    question,
                    retrieved_chunks
                )

            st.subheader("Answer")
            st.write(answer)

            # Remove duplicate page numbers
            source_pages = sorted(
                set(chunk["page"] for chunk in retrieved_chunks)
            )

            st.subheader("Sources")
            st.write(
                "Pages: " +
                ", ".join(str(page) for page in source_pages)
            )

            # Optional: show retrieved evidence
            with st.expander("View retrieved document sections"):
                for chunk in retrieved_chunks:
                    st.markdown(f"**Page {chunk['page']}**")
                    st.write(chunk["text"])
                    st.divider()

        else:
            st.warning("Please enter a question.")