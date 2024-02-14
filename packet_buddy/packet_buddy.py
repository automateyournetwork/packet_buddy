import os
import subprocess
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Message classes
class Message:
    def __init__(self, content):
        self.content = content

class HumanMessage(Message):
    """Represents a message from the user."""
    pass

class AIMessage(Message):
    """Represents a message from the AI."""
    pass

# Define a class for chatting with pcap data
class ChatWithPCAP:
    def __init__(self, json_path):
        self.json_path = json_path
        self.conversation_history = []
        self.load_json()
        self.split_into_chunks()
        self.store_in_chroma()
        self.setup_conversation_memory()
        self.setup_conversation_retrieval_chain()

    def load_json(self):
        self.loader = JSONLoader(
            file_path=self.json_path,
            jq_schema=".[] | ._source.layers",
            text_content=False
        )
        self.pages = self.loader.load_and_split()

    def split_into_chunks(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        self.docs = self.text_splitter.split_documents(self.pages)

    def store_in_chroma(self):
        embeddings = OpenAIEmbeddings()
        self.vectordb = Chroma.from_documents(self.docs, embedding=embeddings)
        self.vectordb.persist()

    def setup_conversation_memory(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def setup_conversation_retrieval_chain(self):
        self.llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")
        self.qa = ConversationalRetrievalChain.from_llm(self.llm, self.vectordb.as_retriever(search_kwargs={"k": 10}), memory=self.memory)

    def chat(self, question):
        response = self.qa.invoke(question)
        self.conversation_history.append(HumanMessage(content=question))
        # Simulating response handling; adjust according to actual response structure
        self.conversation_history.append(AIMessage(content=response.get('answer', 'Response not structured as expected.')))
        return response

# Function to convert pcap to JSON
def pcap_to_json(pcap_path, json_path):
    command = f'tshark -r {pcap_path} -T json > {json_path}'
    subprocess.run(command, shell=True)

# Streamlit UI for uploading and converting pcap file
def upload_and_convert_pcap():
    st.title('Packet Buddy - Chat with Packet Captures')
    uploaded_file = st.file_uploader("Choose a PCAP file", type="pcap")
    if uploaded_file:
        if not os.path.exists('temp'):
            os.makedirs('temp')
        pcap_path = os.path.join("temp", uploaded_file.name)
        json_path = pcap_path + ".json"
        with open(pcap_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        pcap_to_json(pcap_path, json_path)
        st.session_state['json_path'] = json_path
        st.success("PCAP file uploaded and converted to JSON.")
        st.button("Proceed to Chat", on_click=lambda: st.session_state.update({"page": 2}))

# Streamlit UI for chat interface
def chat_interface():
    st.title('Packet Buddy - Chat with Packet Captures')
    json_path = st.session_state.get('json_path')
    if not json_path or not os.path.exists(json_path):
        st.error("PCAP file missing or not converted. Please go back and upload a PCAP file.")
        return

    if 'chat_instance' not in st.session_state:
        st.session_state['chat_instance'] = ChatWithPCAP(json_path=json_path)

    user_input = st.text_input("Ask a question about the PCAP data:")
    if user_input and st.button("Send"):
        with st.spinner('Thinking...'):
            response = st.session_state['chat_instance'].chat(user_input)
            st.markdown("**Answer:**")
            if isinstance(response, dict) and 'answer' in response:
                st.markdown(response['answer'])
            else:
                st.markdown("No specific answer found.")

            st.markdown("**Chat History:**")
            for message in st.session_state['chat_instance'].conversation_history:
                prefix = "*You:* " if isinstance(message, HumanMessage) else "*AI:* "
                st.markdown(f"{prefix}{message.content}")

if __name__ == "__main__":
    if 'page' not in st.session_state:
        st.session_state['page'] = 1

    if st.session_state['page'] == 1:
        upload_and_convert_pcap()
    elif st.session_state['page'] == 2:
        chat_interface()
