import os

#---- Load data environment variables from .env file ----#
from dotenv import load_dotenv
load_dotenv()

#---- Data Ingestion : Read text file and convert to document ----#
from langchain_community.document_loaders import TextLoader
loader = TextLoader("Sample_data.txt")
documents = loader.load()

#---- Data Preprocessing : Split documents into chunks ----#
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

#---- Data Preprocessing : Create embeddings for the document chunks ----#
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embedding model is ready!")

#---- Data Preprocessing : Create a vectorstore from the document chunks and embeddings ----#
from langchain_community.vectorstores import FAISS
vector_store = FAISS.from_documents(docs, embeddings)
print("Vectorstore is ready!")

#--- Data Preprocessing : Create a retriever from the vectorstore ----#
retriever = vector_store.as_retriever(search_kwargs={"k": 2})  # return top 2 results
print("Retriever is ready!")

#--- Data Preprocessing : Create a prompt template for the retriever ----#
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Prompt: a template with a blank {topic} to fill in
prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant. Answer the user's question
using only the context provided below.
Context:
{context}
User Question:
{topic}
If the context does not contain enough information,
say so clearly. Do not invent facts.
Answer in simple English.
""")
# 2. Model: the OpenAI chat model
model = ChatOpenAI(model="gpt-4o-mini")
# 3. Output parser: turn the model's reply into plain text
output_parser = StrOutputParser()
print ("All 3 Prompt, Model and Output parser are ready!")

#--- Data Preprocessing : Create a chain from the prompt, model and output parser ----#
# Create a RAG chain by connecting the retriever, prompt, model and parser
from langchain_core.runnables import RunnablePassthrough
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {
        "context": retriever | format_docs,
        "topic": RunnablePassthrough()
    }
    | prompt
    | model
    | output_parser
)

print("RAG chain is ready!")


#--- User Interaction : Get user input and generate an answer ----#
while True:
    topic = input("Enter your question:or type 'exit' to stop ")
    if topic.lower() == "exit":
        break
    answer = chain.invoke(topic)

    print("\nAI Answer:")
    print(answer)