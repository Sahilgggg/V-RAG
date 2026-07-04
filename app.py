import streamlit as st
import os
from operator import itemgetter
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEndpointEmbeddings
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


st.set_page_config(page_title="YouTube RAG Assistant", page_icon="📺", layout="wide")
st.title("📺 Interactive YouTube Video Assistant")
st.write("Extract insights, generate automatic summaries, and chat directly with any YouTube video.")

def get_video_id(url_or_id):
    if "v=" in url_or_id:
        return url_or_id.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url_or_id:
        return url_or_id.split("youtu.be/")[1].split("?")[0]
    return url_or_id

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
            video_id = get_video_id(video_input)
            
            try: 
                # 1. Fetch Transcripts
                transcript_data = YouTubeTranscriptApi.get_transcript(
                            video_id, 
                            languages=['en'], 
                            cookies="cookies.txt"
                        )
                transcript = " ".join(chunk["text"] for chunk in transcript_data)
                # 2. Chunking
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.create_documents([transcript])
                
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

                # 5. Prompts
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an AI assistant analyzing a video transcript. Answer using ONLY the context below.\n\nContext:\n{context}"),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{question}")
                ])
                
                
                context_and_question = RunnableParallel({
                    'context': itemgetter("question") | retriever | RunnableLambda(format_docs),
                    'question': itemgetter("question")
                })
                
                base_chain = context_and_question | prompt | chat_model | StrOutputParser()
                
                st.session_state.final_chain = RunnableWithMessageHistory(
                    base_chain,
                    lambda session_id: msgs,
                    input_messages_key="question",
                    history_messages_key="history"
                )
                
                summary_generation_chain = context_and_question | ChatPromptTemplate.from_messages([
                    ("system", "Provide a systematic, bulleted summary highlighting the core concepts from this video context.\n\nContext:\n{context}"),
                    ("human", "{question}")
                ]) | chat_model | StrOutputParser()
                
                st.session_state.summary = summary_generation_chain.invoke({"question": "Summarize the video contents structurally."})
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