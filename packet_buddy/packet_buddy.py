import os
import json
import requests
import subprocess
import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import JSONLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

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

# Function to generate priming text based on pcap data
def returnSystemText(pcap_data: str) -> str:
    PACKET_WHISPERER = f"""
        You are a helper assistant specialized in analysing packet captures used for troubleshooting & technical analysis. Use the information present in packet_capture_info to answer all the questions truthfully. If the user asks about a specific application layer protocol, use the following hints to inspect the packet_capture_info to answer the question. Format your response in markdown text with line breaks & emojis.

        hints :
        http means tcp.port = 80
        https means tcp.port = 443
        snmp means udp.port = 161 or udp.port = 162
        ntp means udp.port = 123
        ftp means tcp.port = 21
        ssh means tcp.port = 22
        BGP means tcp.port = 179
        OSPF uses IP protocol 89 (not TCP/UDP port-based, but rather directly on top of IP)
        MPLS doesn't use a TCP/UDP port as it's a data-carrying mechanism for high-performance telecommunications networks
        DNS means udp.port = 53 (also tcp.port = 53 for larger queries or zone transfers)s
        DHCP uses udp.port = 67 for the server and udp.port = 68 for the client
        SMTP means tcp.port = 25 (for email sending)
        POP3 means tcp.port = 110 (for email retrieval)
        IMAP means tcp.port = 143 (for email retrieval, with more features than POP3)
        HTTPS means tcp.port = 443 (secure web browsing)
        LDAP means tcp.port = 389 (for accessing and maintaining distributed directory information services over an IP network)
        LDAPS means tcp.port = 636 (secure version of LDAP)
        SIP means tcp.port = 5060 or udp.port = 5060 (for initiating interactive user sessions involving multimedia elements such as video, voice, chat, gaming, etc.)
        RTP (Real-time Transport Protocol) doesn't have a fixed port but is commonly used in conjunction with SIP for the actual data transfer of audio and video streams.
    """
    # Might be redundant - pcap data - alraedy doing rag - less tokens
    return PACKET_WHISPERER

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
        self.priming_text = self.generate_priming_text()

    def load_json(self):
        self.loader = JSONLoader(
            file_path=self.json_path,
            jq_schema=".[] | ._source.layers",
            text_content=False
        )
        self.pages = self.loader.load_and_split()

    def split_into_chunks(self):
        self.text_splitter = SemanticChunker(FastEmbedEmbeddings())
        self.docs = self.text_splitter.split_documents(self.pages)

    def store_in_chroma(self):
        embeddings = FastEmbedEmbeddings()
        self.vectordb = Chroma.from_documents(self.docs, embedding=embeddings)
        self.vectordb.persist()

    def setup_conversation_memory(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def setup_conversation_retrieval_chain(self):
        self.llm = Ollama(model=st.session_state['selected_model'], base_url="http://ollama:11434")
        self.qa = ConversationalRetrievalChain.from_llm(self.llm, self.vectordb.as_retriever(search_kwargs={"k": 10}), memory=self.memory)

    def generate_priming_text(self):
        pcap_summary = " ".join([str(page) for page in self.pages[:5]])
        return returnSystemText(pcap_summary)

    def chat(self, question):
        # Combine the original question with the priming text
        primed_question = self.priming_text + "\n\n" + question

        response = self.qa.invoke(primed_question)
        
        if response:
            st.write("Query:", primed_question)
            st.write("Response:", response['answer'])

            return {'answer': response['answer']}
   
# Function to convert pcap to JSON
def pcap_to_json(pcap_path, json_path):
    command = f'tshark -nlr {pcap_path} -T json > {json_path}'
    subprocess.run(command, shell=True)

def get_ollama_models(base_url):
    try:       
        response = requests.get(f"{base_url}api/tags")  # Corrected endpoint
        response.raise_for_status()
        models_data = response.json()
        
        # Extract just the model names for the dropdown
        models = [model['name'] for model in models_data.get('models', [])]
        return models
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to get models from Ollama: {e}")
        return []

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
        # Fetch and display the models in a select box
        models = get_ollama_models("http://ollama:11434/")  # Make sure to use the correct base URL
        if models:
            selected_model = st.selectbox("Select Model", models)
            st.session_state['selected_model'] = selected_model
            
            if st.button("Proceed to Chat"):
                st.session_state['page'] = 2        

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
            st.markdown("**Synthesized Answer:**")
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
