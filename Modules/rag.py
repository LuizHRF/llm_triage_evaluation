
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
class rag_agent:
    
    def __init__(self, encoder_model_name='all-MiniLM-L6-v2', protocol_text: str=None):
        self.encoder = SentenceTransformer(encoder_model_name)
        if protocol_text is not None:
            self.chunks = self.txt_to_chunks_marks(protocol_text)
            self.index_base = self.rebuild_index_base(self.chunks)
        else:
            self.chunks = self._pdf_to_text_chunks(pdf_path)
            self.index_base = self.rebuild_index_base(self.chunks)

    def txt_to_chunks(self, text, chunk_size=250):
        chunks = []
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append(chunk)
        return chunks
    
    def txt_to_chunks_marks(self, texts):
        return texts.split("#-#")

    def rebuild_index_base(self, text_chunks):
        embeddings = self.encoder.encode(text_chunks)
        self.index_base = faiss.IndexFlatL2(embeddings.shape[1])
        self.index_base.add(np.array(embeddings))
        return self.index_base

    def _retrieve_docs(self, query, top_k=5):
        if self.index_base is None:
            raise ValueError("Index base not built yet.")
        query_embedding = self.encoder.encode([query])
        D, I = self.index_base.search(np.array(query_embedding), top_k)
        return I[0]
    
    def improve_query(self, query: str, patient_info: str):

        documents = self._retrieve_docs(patient_info.replace("\n", " "), 2)
        documents = [int(i) for i in documents]

        chunks_found = []
        for number in documents:
            chunks_found.append(self.chunks[number])

        if len(chunks_found) == 0:
            return query
        
        new_query = query + "\nConsidere os seguintes trechos do protocolo de triagem como subsídio para sua decisão:"
        for i, doc in enumerate(chunks_found):
            new_query += f"\nTrecho {i+1}:{doc}"
        return new_query

