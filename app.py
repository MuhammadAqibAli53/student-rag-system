import os
import gradio as gr
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import logging
import tempfile
import shutil
from datetime import datetime

# Core libraries for RAG
import faiss
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading and text extraction"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""
    
    def extract_text_from_txt(self, txt_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading text file: {e}")
            return ""
    
    def process_document(self, file_path: str, filename: str) -> List[dict]:
        """Process document and return chunks with metadata"""
        if filename.lower().endswith('.pdf'):
            text = self.extract_text_from_pdf(file_path)
        elif filename.lower().endswith('.txt'):
            text = self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {filename}")
        
        if not text.strip():
            raise ValueError(f"No text extracted from {filename}")
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create chunk objects with metadata
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunks.append({
                'text': chunk,
                'source': filename,
                'chunk_id': i,
                'timestamp': datetime.now().isoformat()
            })
        
        return processed_chunks

class VectorStore:
    """FAISS-based vector store for document embeddings"""
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.chunks = []
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        
    def add_documents(self, chunks: List[dict]) -> int:
        """Add document chunks to the vector store"""
        if not chunks:
            return 0
            
        # Extract text for embedding
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        
        # Initialize or expand FAISS index
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Store chunks metadata
        self.chunks.extend(chunks)
        
        return len(chunks)
    
    def search(self, query: str, k: int = 5) -> List[dict]:
        """Search for relevant chunks"""
        if self.index is None or len(self.chunks) == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), min(k, len(self.chunks)))
        
        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # Valid index
                chunk = self.chunks[idx].copy()
                chunk['similarity_score'] = float(score)
                results.append(chunk)
        
        return results
    
    def clear(self):
        """Clear the vector store"""
        self.index = None
        self.chunks = []
    
    def get_stats(self) -> dict:
        """Get statistics about the vector store"""
        return {
            'total_chunks': len(self.chunks),
            'total_documents': len(set(chunk['source'] for chunk in self.chunks)),
            'index_size': self.index.ntotal if self.index else 0
        }

class RAGSystem:
    """Main RAG system combining retrieval and generation"""
    
    def __init__(self, hf_token: Optional[str] = None):
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        
        # Initialize Hugging Face Inference Client
        # Use a more reliable free model for HF Spaces
        self.llm_client = InferenceClient(
            model="google/flan-t5-base",  # More reliable free model
            token=hf_token
        )
        
        logger.info("RAG System initialized successfully")
    
    def add_document(self, file_path: str, filename: str) -> str:
        """Add a document to the knowledge base"""
        try:
            chunks = self.document_processor.process_document(file_path, filename)
            chunks_added = self.vector_store.add_documents(chunks)
            
            stats = self.vector_store.get_stats()
            return f"✅ Successfully processed '{filename}'\n" \
                   f"📄 Added {chunks_added} chunks\n" \
                   f"📊 Total: {stats['total_chunks']} chunks from {stats['total_documents']} documents"
                   
        except Exception as e:
            error_msg = f"❌ Error processing '{filename}': {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def answer_question(self, question: str, max_chunks: int = 3) -> Tuple[str, str]:
        """Answer a question using RAG pipeline"""
        if not question.strip():
            return "Please ask a question.", ""
        
        # Check if knowledge base is empty
        stats = self.vector_store.get_stats()
        if stats['total_chunks'] == 0:
            return "📚 Please upload some study materials first before asking questions.", ""
        
        try:
            # 1. Retrieve relevant chunks
            relevant_chunks = self.vector_store.search(question, k=max_chunks)
            
            if not relevant_chunks:
                return "🤔 I couldn't find relevant information in your study materials for this question.", ""
            
            # 2. Prepare context from retrieved chunks
            context_parts = []
            sources_info = []
            
            for i, chunk in enumerate(relevant_chunks, 1):
                context_parts.append(f"[Context {i}]: {chunk['text']}")
                sources_info.append(f"📄 **Source {i}**: {chunk['source']} (Chunk {chunk['chunk_id']}, Similarity: {chunk['similarity_score']:.3f})")
            
            context = "\n\n".join(context_parts)
            sources = "\n".join(sources_info)
            
            # 3. Create prompt for LLM
            prompt = f"""Based on the following study materials, please answer the question accurately.

Study Materials:
{context}

Question: {question}

Answer:"""

            # 4. Generate answer using LLM
            try:
                response = self.llm_client.text_generation(
                    prompt,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    return_full_text=False
                )
                
                if isinstance(response, str):
                    answer = response.strip()
                else:
                    answer = response.generated_text.strip() if hasattr(response, 'generated_text') else str(response).strip()
                    
            except Exception as llm_error:
                logger.warning(f"LLM generation failed: {llm_error}")
                # Fallback: provide a structured answer based on retrieved chunks
                answer = f"Based on your study materials, here's what I found:\n\n"
                for i, chunk in enumerate(relevant_chunks, 1):
                    preview = chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
                    answer += f"**From {chunk['source']}:**\n{preview}\n\n"
                answer += "💡 *Note: This is a direct excerpt from your study materials.*"
            
            return answer, sources
            
        except Exception as e:
            error_msg = f"❌ Error answering question: {str(e)}"
            logger.error(error_msg)
            return error_msg, ""
    
    def clear_knowledge_base(self) -> str:
        """Clear all documents from the knowledge base"""
        self.vector_store.clear()
        return "🗑️ Knowledge base cleared successfully!"
    
    def get_knowledge_base_info(self) -> str:
        """Get information about the current knowledge base"""
        stats = self.vector_store.get_stats()
        if stats['total_chunks'] == 0:
            return "📚 Knowledge base is empty. Upload some study materials to get started!"
        
        # Get unique sources
        sources = list(set(chunk['source'] for chunk in self.vector_store.chunks))
        sources_list = "\n".join([f"  • {source}" for source in sources])
        
        return f"""📊 **Knowledge Base Statistics:**
📄 **Documents**: {stats['total_documents']}
🧩 **Total Chunks**: {stats['total_chunks']}
📋 **Files uploaded**:
{sources_list}"""

# Initialize the RAG system with HF token from environment
hf_token = os.environ.get('HF_TOKEN')
rag_system = RAGSystem(hf_token=hf_token)

def upload_file(file):
    """Handle file upload"""
    if file is None:
        return "Please select a file to upload.", rag_system.get_knowledge_base_info()
    
    try:
        filename = os.path.basename(file.name)
        result = rag_system.add_document(file.name, filename)
        return result, rag_system.get_knowledge_base_info()
    except Exception as e:
        error_msg = f"❌ Upload failed: {str(e)}"
        logger.error(error_msg)
        return error_msg, rag_system.get_knowledge_base_info()

def ask_question(question, history):
    """Handle question asking"""
    answer, sources = rag_system.answer_question(question)
    
    # Format response for chat interface
    response = answer
    if sources:
        response += f"\n\n---\n**📚 Sources:**\n{sources}"
    
    history.append((question, response))
    return history, ""

def clear_knowledge_base():
    """Handle knowledge base clearing"""
    result = rag_system.clear_knowledge_base()
    return result, rag_system.get_knowledge_base_info(), []

def refresh_info():
    """Refresh knowledge base information"""
    return rag_system.get_knowledge_base_info()

# Create Gradio interface
with gr.Blocks(
    title="🎓 Student RAG System",
    theme=gr.themes.Soft(),
    css="""
    .main-header { text-align: center; margin-bottom: 2rem; }
    .upload-section { background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
    .chat-section { background: #ffffff; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
    .info-section { background: #e3f2fd; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
    """
) as demo:
    
    # Header
    gr.HTML("""
    <div class="main-header">
        <h1>🎓 Student RAG System</h1>
        <p>Upload your study materials and ask questions to get contextual answers!</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # File Upload Section
            with gr.Group():
                gr.HTML('<div class="upload-section">')
                gr.Markdown("### 📁 Upload Study Materials")
                
                file_input = gr.File(
                    label="Choose PDF or TXT files",
                    file_types=[".pdf", ".txt"],
                    type="filepath"
                )
                
                upload_btn = gr.Button("📤 Upload Document", variant="primary")
                upload_status = gr.Textbox(
                    label="Upload Status",
                    interactive=False,
                    max_lines=5
                )
                gr.HTML('</div>')
            
            # Knowledge Base Info Section  
            with gr.Group():
                gr.HTML('<div class="info-section">')
                gr.Markdown("### 📊 Knowledge Base")
                
                kb_info = gr.Textbox(
                    label="Current Status",
                    value=rag_system.get_knowledge_base_info(),
                    interactive=False,
                    max_lines=10
                )
                
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh", size="sm")
                    clear_btn = gr.Button("🗑️ Clear All", variant="secondary", size="sm")
                
                gr.HTML('</div>')
        
        with gr.Column(scale=2):
            # Chat Section
            with gr.Group():
                gr.HTML('<div class="chat-section">')
                gr.Markdown("### 💬 Ask Questions About Your Study Materials")
                
                chatbot = gr.Chatbot(
                    label="RAG Assistant",
                    height=500,
                    show_label=False,
                    avatar_images=("👤", "🤖")
                )
                
                with gr.Row():
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask a question about your study materials...",
                        scale=4,
                        show_label=False
                    )
                    ask_btn = gr.Button("🔍 Ask", variant="primary", scale=1)
                
                gr.HTML('</div>')
            
            # Examples
            gr.Examples(
                examples=[
                    ["What are the main concepts covered in these materials?"],
                    ["Can you summarize the key points?"],
                    ["What is the definition of [specific term]?"],
                    ["How does [concept A] relate to [concept B]?"],
                    ["What are the important formulas or equations?"]
                ],
                inputs=question_input
            )
    
    # Event handlers
    upload_btn.click(
        fn=upload_file,
        inputs=[file_input],
        outputs=[upload_status, kb_info]
    )
    
    ask_btn.click(
        fn=ask_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot, question_input]
    )
    
    question_input.submit(
        fn=ask_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot, question_input]
    )
    
    clear_btn.click(
        fn=clear_knowledge_base,
        outputs=[upload_status, kb_info, chatbot]
    )
    
    refresh_btn.click(
        fn=refresh_info,
        outputs=[kb_info]
    )
    
    # Footer
    gr.HTML("""
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
        <p><strong>🚀 How to use:</strong></p>
        <ol style="text-align: left; max-width: 600px; margin: 0 auto;">
            <li>Upload your study materials (PDF or TXT files)</li>
            <li>Wait for processing to complete</li>
            <li>Ask questions about your materials</li>
            <li>Get contextual answers with source citations</li>
        </ol>
        <p style="margin-top: 1rem;"><em>Built with ❤️ for students using RAG technology</em></p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch()