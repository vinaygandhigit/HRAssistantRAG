import os
import PyPDF2
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List
import pickle

def read_text_from_file(file_path:str) -> str :
    extension = os.path.splitext(file_path)[1].lower()

    if extension == '.pdf':
        with open(file_path,'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() + '\n'
            return text.strip()
    elif (extension == '.docx' or extension == '.doc') :
        doc = Document(file_path)
        text = '\n'.join([para.text for para in doc.paragraphs])
        return text.strip()
    elif extension == '.txt':
        with open(file_path,'r',encoding='utf-8') as file:
            return file.read().strip()
    else :
        raise ValueError(f"Unsupported file type : {extension}")

def chunk_text(text:str, chunk_size: int=500, chunk_overlap: int=50) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def faiss_index(policies_folder: str, index_path: str='faiss_index.index', chunks_path: str = 'chunks.pkl', chunk_size: int=500, chunk_overlap: int=50):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_dim = model.get_sentence_embedding_dimension()

    index = faiss.IndexFlatL2(embedding_dim)
    all_chunks = []
    documents = []
    for filename in os.listdir(policies_folder):
        file_path = os.path.join(policies_folder, filename)
        if os.path.isfile(file_path):
            try:
                text = read_text_from_file(file_path)
                documents.append(text)
                chunks = chunk_text(text, chunk_size, chunk_overlap)
                all_chunks.extend(chunks)

                embedding = model.encode(chunks)
                index.add(np.array(embedding).astype('float32'))

                print(f"Processed {filename} with {len(chunks)} chunks")
            except Exception as e:
                print(f"Error processing {filename} : {e}")
    
    faiss.write_index(index, index_path)
    
    with open(chunks_path, 'wb') as f:
        pickle.dump(all_chunks, f)
    
    temp_file = "policies.txt"

    training_text = "\n\n".join(documents)
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(training_text)
    
    print(f"FAISS index saved to {index_path}. Chunks saved to {chunks_path}. "
          f"Text saved to {temp_file}. Total chunks: {len(all_chunks)}")
    

def main():
    #script_dir = os.path.dirname(os.path.abspath(__file__))
    #policies_folder = os.path.join(script_dir, "policies")
    policies_folder="policies"
    faiss_index(policies_folder=policies_folder)

if __name__ == "__main__":
    main()                



