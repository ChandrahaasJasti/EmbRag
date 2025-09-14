import faiss
import os
import json
import numpy as np
import requests
from pathlib import Path
from baseclass import VectorStore
from trafilatura import fetch_url, extract
import pymupdf4llm

class FaissVectorStore(VectorStore):
    def __init__(self, docs_path: str, faiss_path: str):
        self.docs_path = docs_path
        self.faiss_path = faiss_path
        
        # Path creation logic
        self._setup_paths()
        self._initialize_files()
        self._load_existing_data()

    def _setup_paths(self):
        """Setup and validate directory paths"""
        # Ensure docs_path exists
        docs_dir = Path(self.docs_path)
        if not docs_dir.exists():
            docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure faiss_path exists
        faiss_dir = Path(self.faiss_path)
        if not faiss_dir.exists():
            faiss_dir.mkdir(parents=True, exist_ok=True)
        
        # Define file paths
        self.cache_path = os.path.join(self.faiss_path, "cache.json")
        self.metadata_path = os.path.join(self.faiss_path, "meta_data.json")
        self.index_path = os.path.join(self.faiss_path, "index.bin")

    def _initialize_files(self):
        """Initialize required files if they don't exist"""
        # Initialize cache.json (empty dict)
        self._ensure_file_exists(self.cache_path, {})
        
        # Initialize meta_data.json (empty list)
        self._ensure_file_exists(self.metadata_path, [])
        
        # Initialize FAISS index if it doesn't exist
        index_file = Path(self.index_path)
        if not index_file.exists():
            self.index = faiss.IndexFlatL2(768)
        else:
            self.index = faiss.read_index(str(index_file))

    def _ensure_file_exists(self, file_path: str, default_content):
        """Ensure a file exists with default content if it doesn't"""
        path = Path(file_path)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(default_content, f, indent=4)

    def _load_existing_data(self):
        """Load existing data from files"""
        # Load cache data
        with open(self.cache_path, 'r') as f:
            self.cache = json.load(f)
        
        # Load metadata
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)

    def chunk_text(self, text: str) -> list[str]:
        """Chunk text into overlapping segments"""
        WORD_COUNT = 512
        OVERLAP = 50
        
        # Split text into words
        words = text.split()
        chunks = []
        
        # Calculate the step size (chunk size - overlap)
        step = WORD_COUNT - OVERLAP
        
        # Create chunks with overlap
        for i in range(0, len(words), step):
            # Get the chunk of words
            chunk = words[i:i + WORD_COUNT]
            
            # Only add chunk if it's not empty
            if chunk:
                # Join words back into text
                chunk_text = ' '.join(chunk)
                chunks.append(chunk_text)
                
                # If we've reached the end of the text, break
                if i + WORD_COUNT >= len(words):
                    break
        
        return chunks

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using Ollama API"""
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": "nomic-embed-text",
                "prompt": text
            }
        )
        response.raise_for_status()
        return np.array(response.json()["embedding"], dtype=np.float32)

    def update_index(self, docs_path: str, cache_path: str) -> list[str]:
        """Process uncached files and update the FAISS index"""
        # Get list of files in docs directory
        files = os.listdir(docs_path)
        
        # Process each file
        for file_name in files:
            if file_name not in self.cache:
                print(f"Processing uncached file: {file_name}")
                
                # Process based on file type
                if (file_name.endswith('.txt') or file_name.endswith('.md')) and not file_name.startswith('url'):
                    self._process_text_file(file_name)
                    
                elif file_name.endswith('.pdf'):
                    self._process_pdf_file(file_name)
                    
                elif file_name.endswith('.txt') and file_name.startswith('url'):
                    self._process_url_file(file_name)
                    
                else:
                    print(f"{file_name} is not a part of [pdf,txt,website] markitdown feature coming soon")
                
                # Mark file as processed in cache
                self.cache[file_name] = "True"
        
        # Save updated cache and metadata
        self._save_data()
        
        # Return list of processed files
        return list(self.cache.keys())

    def _process_text_file(self, file_name: str):
        """Process .txt and .md files"""
        file_path = os.path.join(self.docs_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        chunks = self.chunk_text(text)
        self._add_chunks_to_index(chunks, file_name)

    def _process_pdf_file(self, file_name: str):
        """Process PDF files using pymupdf4llm"""
        file_path = os.path.join(self.docs_path, file_name)
        md_text = pymupdf4llm.to_markdown(file_path)
        chunks = self.chunk_text(md_text)
        self._add_chunks_to_index(chunks, file_name)

    def _process_url_file(self, file_name: str):
        """Process URL files containing web links"""
        file_path = os.path.join(self.docs_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            links = f.read()
            urls = links.split(',')
        
        chunks = []
        for url in urls:
            url = url.strip()  # Remove whitespace
            try:
                downloaded = fetch_url(url)
                result = extract(downloaded)
                if result is None:
                    print(f"Could not access the URL {url} because of authentication")
                else:
                    chunks.append(result)
                    # Add URL content to metadata
                    self.metadata.append({'doc': f"url_{url}", "content": result})
            except Exception as e:
                print(f"Error processing URL {url}: {e}")
        
        self._add_chunks_to_index(chunks, file_name)

    def _add_chunks_to_index(self, chunks: list[str], file_name: str):
        """Add chunks to FAISS index and metadata"""
        if not chunks:
            return
            
        embeds = []
        for k, chunk in enumerate(chunks):
            # Create metadata entry
            metadata_entry = {
                'doc': file_name,
                'id': k,
                'content': chunk
            }
            self.metadata.append(metadata_entry)
            
            # Get embedding
            embedding = self.get_embedding(chunk)
            embeds.append(embedding)
        
        # Add embeddings to FAISS index
        if embeds:
            embeddings_array = np.stack(embeds)
            self.index.add(embeddings_array)

    def _save_data(self):
        """Save cache, metadata, and FAISS index to disk"""
        # Save cache
        with open(self.cache_path, 'w') as f:
            json.dump(self.cache, f, indent=4)
        
        # Save metadata
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)
        
        # Save FAISS index
        faiss.write_index(self.index, self.index_path)

    def add_documents(self, documents: list[str]) -> None:
        pass

    def delete_documents(self, documents: list[str]) -> None:
        pass



class FaissOllamaVectorStore(VectorStore):
    def __init__(self):
        pass

    def connect(self,docs,faiss):
        pass