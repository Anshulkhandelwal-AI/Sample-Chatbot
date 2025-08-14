import os
from uuid import uuid4
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker
from langchain_docling.loader import ExportType

class RAGPipeline:
    def __init__(self, gemini_api_key, pc_api_key, pc_env,
                 pc_index="rag-index-2", embedding_model="all-MiniLM-L6-v2",
                 llm_model="gemini-1.5-flash", temperature=0.3):
        
        genai.configure(api_key=gemini_api_key)
        os.environ["GEMINI_API_KEY"] = gemini_api_key    
        self.pc = Pinecone(api_key=pc_api_key)
        self.embedding = SentenceTransformer(embedding_model)
        self.llm_model = llm_model
        self.temperature = temperature
        self.index_name = pc_index
        self.docs = None

        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(name=self.index_name,
                                 dimension=384,
                                 metric="cosine",
                                 spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        self.index = self.pc.Index(self.index_name)

    def Load_content(self, file_path):
        loader = DoclingLoader(
            file_path=file_path,
            export_type=ExportType.DOC_CHUNKS,
            chunker=HybridChunker(tokenizer="sentence-transformers/all-MiniLM-L6-v2")
        )
        self.docs = loader.load()

    def Embedding_Upsert(self):
        if not self.docs:
            raise ValueError("No documents loaded.")
        
        texts = [doc.page_content for doc in self.docs]
        embeddings = self.embedding.encode(texts, convert_to_numpy=True).tolist()

        to_upsert = []
        for i, emb in enumerate(embeddings):
            metadata = {
                "source": self.docs[i].metadata.get("source", ""),
                "text": self.docs[i].page_content
            }
            vector_id = str(uuid4())
            to_upsert.append((vector_id, emb, metadata))

        self.index.upsert(vectors=to_upsert)

    def Retriever(self, query, k=5):
        q_emb = self.embedding.encode([query])[0].tolist()
        result = self.index.query(vector=q_emb, top_k=k, include_metadata=True)
        return [match.metadata for match in result.matches]

    def Generate_answer(self, query, contexts):
        context = "\n\n".join([con.get("text", "") for con in contexts])
        if not context.strip():
            return "No relevant context found in the document."

        prompt = f"""
        You are an AI assistant. Use the following document context to answer the question.
        Preserve tables/forms structure if applicable.

        Context:
        {context}

        Question:
        {query}

        Answer:
        """
        model = genai.GenerativeModel(self.llm_model)
        response = model.generate_content(prompt)
        return response.text
