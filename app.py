import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import gradio as gr

# Load environment variables
load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

MODEL = "gpt-4o-mini"
db_name = "vector_db"

# Load documents from folder
folders = glob.glob("documents/*")

def add_metadata(doc, doc_type):
    doc.metadata["doc_type"] = doc_type
    return doc

text_loader_kwargs = {'encoding': 'utf-8'}
documents = []
for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    folder_docs = loader.load()
    documents.extend([add_metadata(doc, doc_type) for doc in folder_docs])

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print(f"Loaded and split into {len(chunks)} chunks.")

embeddings = OpenAIEmbeddings()
if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
print(f"Vectorstore created with {vectorstore._collection.count()} documents.")

retriever = vectorstore.as_retriever()
llm = ChatOpenAI(temperature=0.7, model_name=MODEL)
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

def chat(question, history):
    result = conversation_chain.invoke({"question": question})
    return result["answer"]

if __name__ == "__main__":
    print("Launching DocuNova chat interface...")
    # Professional Gradio UI with only title and description above chatbot
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # <span style='color:#3B82F6;font-weight:700'>DocuNova</span>  
            <span style='font-size:1.2em;'>A new way to explore documents</span>
            """,
            elem_id="title"
        )
        chat_ui = gr.ChatInterface(
            chat,
            chatbot=gr.Chatbot(),
            textbox=gr.Textbox(placeholder="Ask a question about your documents..."),
            title=None,
            description=None,
            theme=gr.themes.Soft()
        )
    demo.launch(inbrowser=True)
