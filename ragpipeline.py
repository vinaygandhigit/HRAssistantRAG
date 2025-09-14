import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class HRPolicyRAG:
    def __init__(self, index_path: str = 'faiss_index.index', 
                 chunks_path: str = 'chunks.pkl', 
                 embedder_model: str = 'all-MiniLM-L6-v2', 
                 generator_path: str = './trainedmodelfinetuned'):
        self.embedder = SentenceTransformer(embedder_model)
        self.index = faiss.read_index(index_path)

        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(generator_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(generator_path)
        print("RAG pipeline initialized")

    def retrieve_chunks(self, query:str, k:int = 5) -> list :
        query_embedding = self.embedder.encode([query]).astype("float32")
        distance, indices = self.index.search(query_embedding, k)
        retrieved_chunks = [self.chunks[idx] for idx in indices[0] if idx < len(self.chunks)]
        return retrieved_chunks
    
    def query_response(self, query: str, retrieved_chunks: list, 
                         max_new_tokens: int = 150, temperature: float = 0.7) -> str:
        context = "\n\n".join(retrieved_chunks)
        prompt = f"Context from HR policies:\n{context}\n\nQuestion: {query}\nAnswer:"

        input = self.tokenizer(
            prompt,
            return_tensors = "pt",
            truncation = True,
            max_length = 1024 - max_new_tokens
        )

        outputs = self.model.generate(
            **input,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer_start = response.find("Answer:") + len("Answer:")
        return response[answer_start:].strip()
    
    def query(self, query: str, k: int = 5, max_new_tokens: int = 150, temperature: float = 0.7) -> str:
        retrieved = self.retrieve_chunks(query, k)
        if not retrieved:
            return "No relevant policies found"
        
        return self.query_response(query, retrieved, max_new_tokens, temperature)

def main():
    rag = HRPolicyRAG()
    query = "What is the company's policy on remote work?"
    response = rag.query(query)
    print("Query:", query)
    print("Response:", response)

    print("====================================")
    
    query = "What is the leave policy?"
    response = rag.query(query)
    print("Query:", query)
    print("Response:", response)

if __name__ == "__main__":
    main()