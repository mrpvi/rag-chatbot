import os
import logging
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
PDF_FILE_PATH = "pdfs/ali_profile.pdf"  # replace the own pdf document path
MODEL_NAME = "deepseek-r1:1.5b"

# FastAPI application setup
app = FastAPI()

# Global variables to store embeddings and vector store
doc_processor = None
qa_system = None
app_ready = False  # Flag to check if the app is ready for questions

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

def initialize_processor():
    """Initializes the document processor and loads the PDF."""
    global doc_processor, app_ready
    doc_processor = DocumentProcessor(MODEL_NAME)
    # Process the specific PDF file
    with open(PDF_FILE_PATH, 'rb') as f:
        doc_processor.process_file(f)
    
    app_ready = True  # Set flag to True after processing the PDF

def initialize_qa_system():
    """Initializes the QA system."""
    global qa_system
    qa_system = QuestionAnswerer(MODEL_NAME)

class DocumentProcessor:
    """Handles document processing operations including loading, splitting, and indexing."""
    def __init__(self, model_name: str):
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.vector_store = None

    def process_file(self, file) -> bool:
        """Process file and prepare for question answering."""
        try:
            file_content = file.read()
            file_hash = calculate_file_hash(file_content)

            # Process new file with file path
            documents = self.load_pdf(PDF_FILE_PATH)  # Use file path here
            chunked_documents = self.split_text(documents)
            self.index_documents(chunked_documents)
            return True
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return False

    def load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF file and return documents."""
        try:
            loader = PDFPlumberLoader(file_path)
            documents = loader.load()
            return documents
        except Exception as e:
            logger.error(f"Error loading PDF: {str(e)}")
            raise
    
    def split_text(self, documents: List[Document]) -> List[Document]:
        """Split documents into semantic chunks."""
        try:
            text_splitter = SemanticChunker(self.embeddings)
            return text_splitter.split_documents(documents)
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            raise
    
    def index_documents(self, documents: List[Document]) -> None:
        """Create new in-memory FAISS index with document embeddings."""
        try:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            raise
    
    def retrieve_relevant_docs(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve relevant documents for the given query."""
        try:
            if not self.vector_store:
                raise ValueError("No documents have been indexed yet.")
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}") 
            raise


class QuestionAnswerer:
    """Handles question answering using retrieved documents."""

    def __init__(self, model_name: str):
        self.model = OllamaLLM(model=model_name)
        self.prompt = ChatPromptTemplate.from_template("""
            You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know but don't make up an answer on your own. 

            Use 3 to 4 sentences maximum and keep the answer concise.
            Question: {question} 

            Context: {context} 

            Answer: 
        """)

    def answer_question(self, question: str, documents: List[Document]) -> str:
        """Generate answer based on question and relevant documents."""
        try:
            context = "\n\n".join([doc.page_content for doc in documents])
            chain = self.prompt | self.model
            return chain.invoke({"question": question, "context": context})
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise


def calculate_file_hash(file_content: bytes) -> str:
    """Calculate SHA-256 hash of file content."""
    return hashlib.sha256(file_content).hexdigest()

# Initialize the processor and QA system at server startup
initialize_processor()
initialize_qa_system()

@app.get("/status")
async def check_status():
    """Endpoint to check if the app is ready to accept questions."""
    if app_ready:
        return {"status": "ready", "message": "App is ready to accept questions."}
    else:
        return {"status": "processing", "message": "App is still processing the PDF file."}

@app.post("/ask_question", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Endpoint to ask a question based on pre-loaded PDF."""
    if not app_ready:
        return AnswerResponse(answer="The app is still processing the PDF. Please try again later.")

    try:
        question = request.question
        # Retrieve relevant documents
        related_documents = doc_processor.retrieve_relevant_docs(question)
        # Get the answer from the QA system
        answer = qa_system.answer_question(question, related_documents)
        return AnswerResponse(answer=answer)
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        return AnswerResponse(answer="An error occurred while processing your question.")

if __name__ == "__main__":
    # The FastAPI server will run and handle requests
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
