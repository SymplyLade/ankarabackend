"""
backend/rag_system.py - SIMPLIFIED without memory modules
"""
import os
from dotenv import load_dotenv
from typing import List, Tuple

load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser


class AnkaraPrintRAGSystem:
    def __init__(self, pdf_path: str = None):
        """Initialize SIMPLIFIED RAG system for AnkaraPrint"""
        print("Initializing AnkaraPrint RAG System...")

        self.pdf_path = pdf_path
        self.pdf_loaded = False
        self.vector_db_ready = False
        self.llm_connected = False
        self.total_chunks = 0
        self.session_data = {}

        self.embeddings = self._setup_embeddings()
        # Try to setup LLM but don't fail hard if API key is missing or LLM isn't reachable.
        try:
            self.llm = self._setup_llm()
        except Exception as e:
            print(f"LLM setup warning: {e}")
            self.llm = None
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None

        if pdf_path and os.path.exists(pdf_path):
            self.process_pdf(pdf_path)
        else:
            print("No PDF path provided. Use process_pdf() to load one.")

    def _setup_embeddings(self):
        """Setup local embeddings (free)"""
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print("Local embeddings ready (all-MiniLM-L6-v2)")
            return embeddings
        except Exception as e:
            print(f"Embeddings setup failed: {e}")
            from chromadb.utils import embedding_functions
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )

    def _setup_llm(self):
        """Setup Gemini LLM"""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")

        model_names = [
            "gemini-2.5-flash",
            "gemini-1.5-flash-001",
            "gemini-1.0-pro",
        ]

        for model_name in model_names:
            try:
                print(f"Testing model: {model_name}...")
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=api_key,
                    temperature=0.1,
                    max_output_tokens=1024
                )
                test_response = llm.invoke("Say 'Hello'")
                print(f"Connected to: {model_name}")
                print(f"Test: '{test_response.content}'")
                self.llm_connected = True
                return llm
            except Exception as e:
                print(f"{model_name} failed: {str(e)[:60]}...")
                continue

        raise ValueError("Could not connect to any Gemini model. Check API key.")

    def process_pdf(self, pdf_path: str) -> int:
        """Process PDF and create vector store"""
        try:
            print(f"\nProcessing PDF: {pdf_path}")

            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF not found: {pdf_path}")

            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            print(f"Loaded {len(documents)} pages")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )

            chunks = text_splitter.split_documents(documents)
            self.total_chunks = len(chunks)
            print(f"Created {self.total_chunks} chunks")

            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory="./chroma_db_ankara",
                collection_name="ankaraprint_knowledge"
            )

            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )

            # Build chain only if LLM is available; otherwise leave retriever ready.
            if self.llm:
                self._build_simple_chain()
            else:
                print("LLM not connected - vectorstore ready, but RAG chain not built.")
            self.pdf_loaded = True
            self.vector_db_ready = True

            print("PDF processing complete!")
            return self.total_chunks

        except Exception as e:
            print(f"PDF processing failed: {e}")
            raise

    def _build_simple_chain(self):
        """Build a SIMPLE RAG chain without memory"""
        prompt_template = """
        You are "AnkaraPrint Assistant", an AI representing the AnkaraPrint project.

        CONTEXT FROM ANKARAPRINT KNOWLEDGE BASE:
        {context}

        USER QUESTION: {question}

        INSTRUCTIONS:
        1. Answer using ONLY the context above
        2. Be professional, accurate, and helpful
        3. If information isn't in context, say: "I don't have that specific information in AnkaraPrint knowledge base"
        4. Always invite further questions

        ANSWER:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)

        def format_docs(docs):
            formatted = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', 'PDF')
                page = doc.metadata.get('page', 'N/A')
                content = doc.page_content
                formatted.append(f"[Source {i}: {source}, Page {page}]\n{content}")
            return "\n\n".join(formatted)

        self.rag_chain = (
            RunnableParallel(
                context=self.retriever | format_docs,
                question=RunnablePassthrough()
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )

        print("Built simple RAG chain (no memory modules needed)")

    def get_response(self, question: str, session_id: str = "default") -> Tuple[str, List[str]]:
        """Get response for a question - SIMPLIFIED"""
        if not self.rag_chain:
            return ("RAG system not ready - either no PDF loaded or LLM unavailable.", [])

        print(f"\nProcessing: '{question[:50]}...'")
        try:
            retrieved_docs = self.retriever.invoke(question)
            sources = [
                f"Page {doc.metadata.get('page', 'N/A')}: {doc.page_content[:80]}..."
                for doc in retrieved_docs
            ]

            response = self.rag_chain.invoke(question)
            print(f"Response generated ({len(response)} chars)")
            return response, sources

        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error: {str(e)}", []

    def get_system_status(self):
        return {
            "status": "online",
            "pdf_loaded": self.pdf_loaded,
            "vector_db_ready": self.vector_db_ready,
            "llm_connected": self.llm_connected,
            "total_chunks": self.total_chunks,
            "vector_db_path": "./chroma_db_ankara"
        }

    def clear_data(self):
        self.session_data = {}
        print("Session data cleared")


def create_default_rag():
    """Create RAG system with default PDF if exists"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    default_pdf = os.path.join(BASE_DIR, "Ankara Printing.pdf")

    if os.path.exists(default_pdf):
        print(f"Found default PDF: {default_pdf}")
        return AnkaraPrintRAGSystem(default_pdf)
    else:
        print(f"Default PDF not found: {default_pdf}")
        print("Use process_pdf() method later to load your PDF.")
        return AnkaraPrintRAGSystem()
