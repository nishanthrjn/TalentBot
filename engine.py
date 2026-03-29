import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# Load variables from .env
load_dotenv()

# Initialize the "Translator" (Must be the same model used in ingest.py)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def ask_talent_bot(question):
    # 1. Load the Memory
    # we use allow_dangerous_deserialization because we created the file ourselves
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # 2. Search the index for the 3 most relevant paragraphs
    docs = db.similarity_search(question, k=3)
    context = "\n".join([d.page_content for d in docs])

    # 3. Setup the LLM (Llama 3 on Groq)
    llm = ChatGroq(
        temperature=0.1, # Keep it very factual
        model_name="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # 4. The Professional Persona
    system_prompt = f"""
    You are TalentBot, the official AI assistant for Nishanth Rajan.
    Nishanth is a Software Engineer with 15+ years of experience.
    
    Rules:
    - Use the context provided below to answer.
    - If the answer isn't in the context, say: "I don't have that specific information, but you can reach Nishanth at nishanthrajandev@gmail.com."
    - Be professional, technical, and concise.
    
    Context:
    {context}
    """

    # 5. Get the Answer
    response = llm.invoke([
        ("system", system_prompt),
        ("user", question)
    ])
    
    return response.content

if __name__ == "__main__":
    # Test it immediately in the terminal!
    query = "What is Nishanth's experience with SOLIDWORKS?"
    print(f"\nSearching for: {query}...")
    print("-" * 30)
    print(ask_talent_bot(query))