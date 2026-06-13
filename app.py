import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ==========================================
# 1. CONFIGURATION & API SETUP
# ==========================================
# IMPORTANT: DO NOT UPLOAD YOUR REAL KEY TO GITHUB! 
os.environ["GOOGLE_API_KEY"] = "AIzaSyC05vN3mSpJUkVuzBxYeROqSUVoH6zlEWE"

st.set_page_config(page_title="Amity Assistant", page_icon="🎓")
st.title("🎓 Amity University Online FAQ Assistant")
st.caption("Ask questions based directly on the college's official brochures and guidelines.")

# The sidebar goes completely on its own below the title!
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8074/8074804.png", width=100)
    st.header("About This Bot")
    st.write("This is an AI-powered admission assistant built using Retrieval-Augmented Generation (RAG).")
    st.divider()
    st.caption("Capabilities:")
    st.caption("✔️ Reads from official PDFs")
    st.caption("✔️ Reads live University URLs")
    st.caption("✔️ Memory-aware interactions")
    st.caption("✔️ Hallucination-free responses")

# ==========================================
# 2. KNOWLEDGE BASE SETUP (CACHED)
# ==========================================
@st.cache_resource
def initialize_knowledge_base():
    loaded_raw_documents = []
    
    # --- A. Load Local PDFs ---
    for filename in os.listdir("data"):
        if filename.endswith(".pdf"):
            try:
                file_path = os.path.join("data", filename)
                loader = PyPDFLoader(file_path)
                loaded_raw_documents.extend(loader.load())
            except Exception as e:
                print(f"Skipping unreadable file: {filename}")
                
    # --- B. Load Live Websites ---
    target_urls = [
        "https://amityonline.com/faq",
        "https://amityonline.com/bca",
        "https://amityonline.com/",
        "https://amityonline.com/bachelor-of-computer-applications-online"  # You can add more Amity URLs here
    ]
    
    for url in target_urls:
        try:
            print(f"Reading website: {url}...")
            web_loader = WebBaseLoader(url)
            loaded_raw_documents.extend(web_loader.load())
        except Exception as e:
            print(f"⚠️ Could not read website: {url}")
                
    # --- C. Process and Embed Data ---
    document_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    processed_text_chunks = document_splitter.split_documents(loaded_raw_documents)
    
    # Use Local Embeddings to completely bypass API rate limits
    ai_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Store in Chroma Vector Database
    vector_store = Chroma.from_documents(documents=processed_text_chunks, embedding=ai_embeddings)
    return vector_store.as_retriever()

document_retriever = initialize_knowledge_base()

# ==========================================
# 3. AI MODEL & MEMORY INITIALIZATION
# ==========================================
generative_ai_model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

# Initialize Session State to remember the conversation
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Re-render past messages on the screen
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

# ==========================================
# 4. PROMPT ENGINEERING & RAG PIPELINE
# ==========================================
context_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given the chat history and the latest user question, formulate a standalone question that can be understood without the history. Do NOT answer it, just reformulate it."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    generative_ai_model, document_retriever, context_prompt
)

system_guardrail_instructions = (
    "You are a polite, formal, and authoritative college campus admission assistant. "
    "If the user says a basic greeting (like 'Hi','Hey', 'Hello', or 'Good morning'), warmly introduce yourself "
    "and ask how you can assist them with BCA admissions. "
    "For all other questions, your objective is to answer strictly using the provided context information. "
    "If a specific college-related answer cannot be extracted from the context below, reply exactly with: "
    "'I am sorry, but I do not have verified information regarding that query in my database.' "
    "Do not hallucinate, speculate, or make up facts outside the text.\n\n"
    "Context Information:\n{context}"
)

prompt_layout = ChatPromptTemplate.from_messages([
    ("system", system_guardrail_instructions),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

document_assembly_chain = create_stuff_documents_chain(generative_ai_model, prompt_layout)
rag_execution_chain = create_retrieval_chain(history_aware_retriever, document_assembly_chain)

# ==========================================
# 5. USER INTERACTION LOOP
# ==========================================
user_chat_query = st.chat_input("Ask a question about programs, fees, or requirements...")

if user_chat_query:
    st.chat_message("user").write(user_chat_query)
    
    with st.spinner("Searching official documents and websites..."):
        pipeline_output = rag_execution_chain.invoke({
            "input": user_chat_query,
            "chat_history": st.session_state.chat_history
        })
    
    final_answer = pipeline_output["answer"]
    st.chat_message("assistant").write(final_answer)
    
    st.session_state.chat_history.extend([
        HumanMessage(content=user_chat_query),
        AIMessage(content=final_answer)
    ])