from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
from anthropic import Anthropic
from dotenv import load_dotenv
import os
from google import genai

load_dotenv()

claude_client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

def extract_text_from_pdf(pdf_file):
    """
    Extract text from a PDF while keeping track of page numbers.
    """

    reader = PdfReader(pdf_file)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text:
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages

def create_chunks(pages, chunk_size=1000, overlap=200):
    """
    Split extracted PDF text into smaller overlapping chunks.
    """

    chunks = []

    for page in pages:
        text = page["text"]
        page_number = page["page"]

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append({
                "text": chunk_text,
                "page": page_number
            })

            start += chunk_size - overlap

    return chunks

def create_embeddings(chunks):
    """
    Convert text chunks into numerical embeddings.
    """

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.encode(texts)

    return embeddings

def store_in_chromadb(chunks, embeddings):
    """
    Store text chunks and their embeddings in ChromaDB.
    """

    client = chromadb.Client()

    collection = client.get_or_create_collection(
        name="documents"
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    texts = [chunk["text"] for chunk in chunks]

    metadatas = [
        {"page": chunk["page"]}
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    return collection

def retrieve_relevant_chunks(question, collection, top_k=5):
    """
    Retrieve the most relevant chunks for the user's question.
    """

    question_embedding = embedding_model.encode([question])

    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=top_k
    )

    retrieved_chunks = []

    for text, metadata in zip(
        results["documents"][0],
        results["metadatas"][0]
    ):
        retrieved_chunks.append({
            "text": text,
            "page": metadata["page"]
        })

    return retrieved_chunks

def generate_answer(question, retrieved_chunks):
    """
    Generate an answer using Gemini based only on retrieved PDF content.
    """

    context = "\n\n".join(
        [
            f"Page {chunk['page']}:\n{chunk['text']}"
            for chunk in retrieved_chunks
        ]
    )

    prompt = f"""
You are a document question-answering assistant.

Answer the question using ONLY the information contained in the
retrieved document context below.

The answer does not need to appear word-for-word in the document.
You may summarize and combine relevant information from the retrieved
context, but do not add outside knowledge.

When possible, mention the page number supporting each point.

If the retrieved context truly contains no relevant information,
say: "The retrieved document context does not contain enough information
to answer this question."

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

Provide a clear and concise answer based on the retrieved context.
"""

    response = gemini_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text

if __name__ == "__main__":

    pdf_path = "documents/rag_survey.pdf"

    print("1. Reading PDF...")
    pages = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(pages)} pages")

    print("2. Creating chunks...")
    chunks = create_chunks(pages)
    print(f"Created {len(chunks)} chunks")

    print("3. Creating embeddings...")
    embeddings = create_embeddings(chunks)
    print(f"Created {len(embeddings)} embeddings")

    print("4. Storing in ChromaDB...")
    collection = store_in_chromadb(chunks, embeddings)

    question = "What are the main challenges and limitations of RAG?"

    print("\n5. Searching for relevant information...")
    results = retrieve_relevant_chunks(
        question,
        collection
    )

    print("\nQUESTION:")
    print(question)

    print("\nRETRIEVED RESULTS:")

    for result in results:
        print("\n-----------------------------")
        print(f"Page: {result['page']}")
        print(result["text"])

    print("\n6. Generating answer with Gemini...")

    answer = generate_answer(
    question,
    results
    )

    print("\nGEMINI ANSWER:")
    print(answer)

    print("\nSOURCES:")
    for result in results:
        print(f"Page {result['page']}")