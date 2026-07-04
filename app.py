import streamlit as st
import os
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEndpointEmbeddings
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from operator import itemgetter

st.set_page_config(page_title="YouTube RAG Assistant", page_icon="📺", layout="wide")
st.title("📺 Interactive YouTube Video Assistant")
st.write("Extract insights, generate automatic summaries, and chat directly with any YouTube video.")

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

msgs = StreamlitChatMessageHistory(key="chat_history")

if "summary" not in st.session_state:
    st.session_state.summary = None
if "final_chain" not in st.session_state:
    st.session_state.final_chain = None

with st.sidebar:
    st.header("Configuration")
    hf_api_token = st.text_input("Hugging Face Access Token", type="password")
    video_input = st.text_input("YouTube Video URL or ID")
    process_btn = st.button("Process & Summarize")
    
    if st.button("Clear Chat History"):
        msgs.clear()
        st.session_state.summary = None
        st.session_state.final_chain = None
        st.rerun()

if process_btn:
    if not hf_api_token:
        st.sidebar.error("Please provide a valid Hugging Face Access Token.")
    elif not video_input:
        st.sidebar.error("Please enter a YouTube link or ID.")
    else:
        with st.spinner("Processing video transcript and building memory..."):
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_token
            
            # Auto-format the URL if the user just pastes an ID like 'aircAruvnKk'
            if "youtube.com" not in video_input and "youtu.be" not in video_input:
                video_url = f"https://www.youtube.com/watch?v={video_input}"
            else:
                video_url = video_input
            
            try:
                # 1. Fetch Transcripts
                loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False)
                docs = loader.load()
                
                # 2. Chunking
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(docs)
                
                if not chunks:
                    st.sidebar.error("No transcript found for this video.")
                    st.stop()
                
                # 3. Embeddings & FAISS
                embeddings = HuggingFaceEndpointEmbeddings(
                    model="sentence-transformers/all-MiniLM-L6-v2",
                    task="feature-extraction",
                    huggingfacehub_api_token=hf_api_token
                )
                vector_store = FAISS.from_documents(chunks, embeddings)
                retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
                
                # 4. LLM Setup
                llm = HuggingFaceEndpoint(
                    repo_id="Qwen/Qwen2.5-7B-Instruct", 
                    task="text-generation",
                    max_new_tokens=512,
                    huggingfacehub_api_token=hf_api_token
                )
                chat_model = ChatHuggingFace(llm=llm)

                # 5. RAG Prompts & Chains
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an AI assistant analyzing a video transcript. Answer using ONLY the context below.\n\nContext:\n{context}"),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{question}")
                ])

                # Use RunnablePassthrough.assign to keep 'question' and 'history', while adding 'context'
                context_and_question = RunnablePassthrough.assign(
                    context=itemgetter("question") | retriever | RunnableLambda(format_docs)
                )
                
                base_chain = context_and_question | prompt | chat_model | StrOutputParser()
                
                st.session_state.final_chain = RunnableWithMessageHistory(
                    base_chain,
                    lambda session_id: msgs,
                    input_messages_key="question",
                    history_messages_key="history"
                )
                
                # 6. Improved Summary Logic (Reads the first 3000 chars of the actual transcript)
                full_text = " ".join([doc.page_content for doc in docs])
                summary_text = full_text[:3000] # Truncate to avoid context window limits
                
                summary_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Provide a systematic, bulleted summary highlighting the core concepts from this video context.\n\nContext:\n{context}"),
                    ("human", "{question}")
                ])
                
                summary_chain = summary_prompt | chat_model | StrOutputParser()
                st.session_state.summary = summary_chain.invoke({
                    "context": summary_text, 
                    "question": "Summarize the video contents structurally."
                })
                
                st.sidebar.success("Processing complete!")
                
            except Exception as e:
                st.sidebar.error(f"Error processing video: {str(e)}")

if st.session_state.summary:
    st.subheader("📋 Systematic Video Summary")
    st.markdown(st.session_state.summary)
    st.markdown("---")

st.subheader("💬 Chat with the Video")

for msg in msgs.messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

if user_query := st.chat_input("Ask something about the video..."):
    if not st.session_state.final_chain:
        st.error("Please enter your API token, a video link, and press 'Process & Summarize' first.")
    else:
        with st.chat_message("user"):
            st.markdown(user_query)
            
        with st.chat_message("assistant"):
            with st.spinner("Analyzing transcript context..."):
                os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_token
                config = {"configurable": {"session_id": "any_static_session_id"}}
                response = st.session_state.final_chain.invoke({"question": user_query}, config=config)
                st.markdown(response)