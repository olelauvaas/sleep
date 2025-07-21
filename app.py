import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

# 🔐 API-nøkkel fra Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key

# 🤖 Modell: Rolig, kunnskapsrik og vennlig
llm = ChatOpenAI(model="gpt-4o", temperature=0.6)

# 📁 Laster alle .txt-filer fra /Data-mappen
@st.cache_resource
def build_vector_db():
    documents = []
    folder_path = "data"

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Mappen '{folder_path}' finnes ikke. Dobbeltsjekk at du har laget den.")

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            path = os.path.join(folder_path, filename)
            try:
                loader = TextLoader(path, encoding='utf-8')
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                st.warning(f"⚠️ Kunne ikke laste {filename}: {e}")

    if not documents:
        raise ValueError("🚫 Ingen gyldige dokumenter ble funnet i 'Data'-mappen.")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_documents(texts, embeddings)
    return db

# 🧠 Matthew Walker-stil (vennlig, pedagogisk, vitenskapelig)
sleep_prompt = PromptTemplate.from_template("""
Du er Matthew Walker – en verdensledende søvnekspert og forfatter av boka "Why We Sleep". 
Svar på en forståelig, rolig og engasjerende måte som hjelper folk å forstå viktigheten av søvn. 
Du bruker gjerne eksempler fra vitenskap, hverdagen og egne foredrag, og forklarer komplekse ting enkelt – som en dyktig formidler ville gjort på TV eller YouTube.

Bruk gjerne metaforer, korte historier eller analogier – men vær alltid korrekt og tydelig.

Spørsmål:
{question}

Relevant info fra dokumentene:
{context}
""")

# 🎛️ Streamlit-oppsett
st.set_page_config(page_title="Søvnråd fra Matthew Walker", page_icon="💤")
st.image("logo.png", width=300)
st.title("Søvnråd fra Matthew Walker")

# 🔍 Spørsmål fra bruker
query = st.text_input("Hva lurer du på om søvn?", placeholder="F.eks. Hvorfor er REM-søvn viktig?")

# 🤖 Svar hvis bruker skriver noe
if query:
    with st.spinner("Jeg tenker meg om..."):
        db = build_vector_db()
        retriever = db.as_retriever()
        context_docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in context_docs])

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": sleep_prompt}
        )

        svar = chain.run(query)
        st.markdown(svar)