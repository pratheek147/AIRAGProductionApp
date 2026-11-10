from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

# Initialize Gemini embedding model (defaults to text embedding model)
embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-001")
EMBEDDING_DIM = 768  # text-embedding-004 outputs 768 dimensions

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200) # characters

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [doc.text for doc in docs if getattr(doc, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    # Use LlamaIndex's batch embedding method for Google GenAI
    return embed_model.get_text_embedding_batch(texts)