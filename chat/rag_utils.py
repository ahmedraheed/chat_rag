# chat/rag_utils.py
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from django.conf import settings

# Use OpenAI API key from settings
def create_vector_store():
    loader = TextLoader("chat/docs/your_doc.txt")  # Your document path here
    documents = loader.load()

    embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("chat/faiss_index")

# chat/rag_utils.py

from langchain_community.vectorstores import FAISS

def query_rag(prompt):
    embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)

    # Allow dangerous deserialization (use only if the file is trusted)
    db = FAISS.load_local("chat/faiss_index", embeddings, allow_dangerous_deserialization=True)

    retriever = db.as_retriever()
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    result = qa_chain.run(prompt)
    return result

