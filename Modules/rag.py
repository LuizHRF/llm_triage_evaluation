#============================================================================================

## Transformar todo o protocolo em texto (Incluindo imagens, fluxogramas, tabelas, etc.)

# ou

## Transformar parcialmente o protocolo em texto (Recuperar apenas o que é texto, tratando de forma básica, 
## e abrir mão dos elementos visuais)

## Vetorizar o pdf

## Criar base de dados vetorizados

## No moemnto da query: Vetorizar a query e buscar por elementos relevantes relacionados na base de dados

## Recosntruir o prompt com as informações recuperadas

## Seguir o fluxo de query no LLM

## Adcionalmente, incorporar na resposta os pedações recuperados por RAG

#============================================================================================

import fitz 
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
class rag_agent:

    """
        RAG Agent for enhancing queries with relevant document references from a PDF.

        Attributes:
            encoder (SentenceTransformer): Model for encoding text into embeddings.
            chunks (list): List of text chunks extracted from the PDF.
            index_base (faiss.IndexFlatL2): FAISS index for efficient similarity search.
    
        Methods:
            _pdf_to_text_chunks(pdf_path, chunk_size): Extracts and chunks text from the PDF.
            rebuild_index_base(text_chunks): Builds the FAISS index from text chunks.
            _retrieve_docs(query, top_k): Retrieves top_k relevant document indices for a given query.
            improve_query(query): Enhances the query by appending relevant document texts.
    
    """
    
    def __init__(self, encoder_model_name='all-MiniLM-L6-v2', pdf_path: str=None, protocol_text: str=None):
        self.encoder = SentenceTransformer(encoder_model_name)
        if protocol_text is not None:
            self.chunks = self.txt_to_chunks([protocol_text])
            self.index_base = self.rebuild_index_base(self.chunks)
        else:
            self.chunks = self._pdf_to_text_chunks(pdf_path)
            self.index_base = self.rebuild_index_base(self.chunks)

    def txt_to_chunks(self, texts, chunk_size=512):
        chunks = []
        for text in texts:
            words = text.split()
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i+chunk_size])
                chunks.append(chunk)
        return chunks

    def _pdf_to_text_chunks(self, pdf_path, chunk_size=512):
        doc = fitz.open(pdf_path)
        pages_to_remove = [0, 1, 2, 3, 4]
        for page_num in sorted(pages_to_remove, reverse=True):
            doc.delete_page(page_num)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append(chunk)
        return chunks

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
    
    def improve_query(self, query: str):
        documents = self._retrieve_docs(query, 3)
        new_query = query + "\nConsidere os seguintes trechos do protocolo como subsídio para sua decisão:\n" + "\n".join(self.chunks[i] for i in documents)
        return new_query

