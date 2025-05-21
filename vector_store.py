import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class VectorStoreManager:
    def __init__(self, persist_directory="vector_db", collection_name="rag_collection_default"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name 

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True
        )
        
        os.makedirs(self.persist_directory, exist_ok=True)
    
    def create_or_update_vector_store(self, documents_with_text_and_meta):
        all_chunks = []
        all_metadatas = []

        for i, doc_data in enumerate(documents_with_text_and_meta):
            chunks_from_doc = self.text_splitter.split_text(doc_data['text'])
            metadatas_from_doc = [{
                "name": doc_data['name'], 
                "type": doc_data['type'],
                "doc_index": i,
            } for _ in chunks_from_doc]
            
            all_chunks.extend(chunks_from_doc)
            all_metadatas.extend(metadatas_from_doc)

        if not all_chunks:
            raise ValueError("No text chunks generated. Cannot create vector store.")
        
        vector_store = Chroma.from_texts(
            texts=all_chunks,
            embedding=self.embedding_model,
            metadatas=all_metadatas,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name 
        )
        
        vector_store.persist()
        return vector_store
    
    def get_vector_store(self):
        if not os.path.exists(self.persist_directory):
             return None
        
        try:
            vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model,
                collection_name=self.collection_name
            )
            if vector_store._collection.count() == 0:
                return None 
            return vector_store
        except Exception as e:
            return None

    def delete_collection(self):
        print(f"Placeholder: Deleting collection '{self.collection_name}' would require Chroma client.")