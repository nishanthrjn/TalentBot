import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_db():
    # 1. Load your resume data
    if not os.path.exists("data/resume.txt"):
        print("❌ Error: data/resume.txt not found!")
        return

    loader = TextLoader("data/resume.txt")
    documents = loader.load()

    # 2. Split text into chunks 
    # (AI works better with small pieces of info rather than one giant block)
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # 3. Initialize Embeddings (Totally Free)
    # This model converts your words into numbers the AI understands
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Create the Vector Store (The Brain)
    print("⏳ Creating 'Brain' index... hang tight.")
    vector_db = FAISS.from_documents(docs, embeddings)

    # 5. Save it locally so you don't have to re-run this every time
    vector_db.save_local("faiss_index")
    print("✅ Success! 'faiss_index' folder created. TalentBot is now trained.")

if __name__ == "__main__":
    create_vector_db()