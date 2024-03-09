import os
import json
from langchain.load import dumps, loads
import subprocess
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_experimental.text_splitter import SemanticChunker

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
        self.text_splitter = SemanticChunker(OpenAIEmbeddings())
        self.docs = self.text_splitter.split_documents(self.pages)

    def store_in_chroma(self):
        embeddings = OpenAIEmbeddings()
        self.vectordb = Chroma.from_documents(self.docs, embedding=embeddings)
        self.vectordb.persist()

    def setup_conversation_memory(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def setup_conversation_retrieval_chain(self):
        self.llm = ChatOpenAI(temperature=0.7, model="gpt-4-1106-preview")
        self.qa = ConversationalRetrievalChain.from_llm(self.llm, self.vectordb.as_retriever(search_kwargs={"k": 10}), memory=self.memory)

    def generate_priming_text(self):
        pcap_summary = " ".join([str(page) for page in self.pages[:5]])
        return returnSystemText(pcap_summary)

    def chat(self, question):
        # Combine the original question with the priming text
        primed_question = self.priming_text + "\n\n" + question

        # Generate related queries based on the primed question
        related_queries_dicts = self.generate_related_queries(primed_question)
        related_queries = [q['query'] for q in related_queries_dicts]

        # Priming each related query individually
        primed_related_queries = [(self.priming_text + "\n\n" + rq) for rq in related_queries]

        # Include the initially primed question as the first query
        queries = [primed_question] + primed_related_queries

        all_results = []

        for query_text in queries:
            st.write(query_text)
            response = None
            if self.llm:
                response = self.qa.invoke(query_text)
            elif self.llm_anthropic:
                response = self.anthropic_qa.invoke(query_text)

            if response:
                all_results.append({'query': query_text, 'answer': response['answer']})
                st.write("Query:", query_text)
                st.write("Response:", response['answer'])

        # After gathering all results, let's ask the LLM to synthesize a comprehensive answer
        if all_results:
            # Assuming reciprocal_rank_fusion is correctly applied and scored_results is prepared
            reranked_results = self.reciprocal_rank_fusion(all_results)
            st.write(reranked_results)
            # Prepare scored_results, ensuring it has the correct structure
            scored_results = [{'score': res['score'], **res['doc']} for res in reranked_results]
            synthesis_prompt = self.create_synthesis_prompt(question, scored_results)
            synthesized_response = self.llm.invoke(synthesis_prompt)
            
            if synthesized_response:
                # Assuming synthesized_response is an AIMessage object with a 'content' attribute
                st.write(synthesized_response)
                final_answer = synthesized_response.content
            else:
                final_answer = "Unable to synthesize a response."
            
            # Update conversation history with the original question and the synthesized answer
            self.conversation_history.append(HumanMessage(content=question))
            self.conversation_history.append(AIMessage(content=final_answer))

            return {'answer': final_answer}
        else:
            self.conversation_history.append(HumanMessage(content=question))
            self.conversation_history.append(AIMessage(content="No answer available."))
            return {'answer': "No results were available to synthesize a response."}

    def create_synthesis_prompt(self, original_question, all_results):
        # Sort the results based on RRF score if not already sorted; highest scores first
        sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)
        st.write("Sorted Results", sorted_results)
        prompt = f"Based on the user's original question: '{original_question}', here are the answers to the original and related questions, ordered by their relevance (with RRF scores). Please synthesize a comprehensive answer focusing on answering the original question using all the information provided below:\n\n"
        
        # Include RRF scores in the prompt, and emphasize higher-ranked answers
        for idx, result in enumerate(sorted_results):
            prompt += f"Answer {idx+1} (Score: {result['score']}): {result['answer']}\n\n"
        
        prompt += "Given the above answers, especially considering those with higher scores, please provide the best possible composite answer to the user's original question."
        
        return prompt

    def generate_related_queries(self, primed_question):
        prompt = f"In light of the original inquiry: '{primed_question}', let's delve deeper and broaden our exploration. Please construct a JSON array containing four distinct but interconnected search queries. Each query should reinterpret the original prompt's essence, introducing new dimensions or perspectives to investigate. Aim for a blend of complexity and specificity in your rephrasings, ensuring each query unveils different facets of the original question. This approach is intended to encapsulate a more comprehensive understanding and generate the most insightful answers possible. Only respond with the JSON array itself."
        response = self.llm.invoke(input=prompt)

        if hasattr(response, 'content'):
            # Directly access the 'content' if the response is the expected object
            generated_text = response.content
        elif isinstance(response, dict) and 'content' in response:
            # Extract 'content' if the response is a dict
            generated_text = response['content']
        else:
            # Fallback if the structure is different or unknown
            generated_text = str(response)
            st.error("Unexpected response format.")

        st.write("Response content:", generated_text)

        # Assuming the 'content' starts with "content='" and ends with "'"
        # Attempt to directly parse the JSON part, assuming no other wrapping
        try:
            json_start = generated_text.find('[')
            json_end = generated_text.rfind(']') + 1
            json_str = generated_text[json_start:json_end]
            related_queries = json.loads(json_str)
            st.write("Parsed related queries:", related_queries)
        except (ValueError, json.JSONDecodeError) as e:
            st.error(f"Failed to parse JSON: {e}")
            related_queries = []

        return related_queries

    def retrieve_documents(self, query):
        # Example: Convert query to embeddings and perform a vector search in ChromaDB
        query_embedding = OpenAIEmbeddings()  # Assuming SemanticChunker can embed text
        search_results = self.vectordb.search(query_embedding, top_k=5)  # Adjust based on your setup
        document_ids = [result['id'] for result in search_results]  # Extract document IDs from results
        return document_ids
    
    def reciprocal_rank_fusion(self, results, k=60):
        fused_scores = {}
        for idx, item in enumerate(results):
            answer_id = item['query']  # Assuming this is unique
            if answer_id not in fused_scores:
                fused_scores[answer_id] = {'score': 0, 'item': item}
            fused_scores[answer_id]['score'] += 1 / (idx + 1 + k)

        reranked_items = sorted(fused_scores.values(), key=lambda x: x['score'], reverse=True)
        return [{'score': item['score'], 'doc': item['item']} for item in reranked_items]  # Adjusted to include scores

# Function to convert pcap to JSON
def pcap_to_json(pcap_path, json_path):
    command = f'tshark -nlr {pcap_path} -T json > {json_path}'
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
