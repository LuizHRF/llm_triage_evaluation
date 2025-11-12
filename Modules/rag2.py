
# https://docs.langchain.com/oss/python/langchain/knowledge-base

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_core.vectorstores import InMemoryVectorStore

def create_vector_store(protocol_path: str, splits: tuple = None) -> tuple[InMemoryVectorStore, HuggingFaceEmbeddings]:
    loader = PyPDFLoader(protocol_path)

    docs = loader.load()
    if splits:
        #15 à 37
        docs = docs[splits[0]:splits[1]]

    # print(len(docs))
    # print(f"{docs[-1].page_content}\n")
    # print(docs[-1].metadata)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=250, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    #print(len(all_splits))

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    vector_1 = embeddings.embed_query(all_splits[0].page_content)
    vector_2 = embeddings.embed_query(all_splits[1].page_content)

    assert len(vector_1) == len(vector_2)
    #print(f"Generated vectors of length {len(vector_1)}\n")

    vector_store = InMemoryVectorStore(embeddings)
    ids = vector_store.add_documents(documents=all_splits)

    return vector_store, embeddings