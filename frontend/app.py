"""V-RAG Frontend -- Streamlit UI that calls the FastAPI backend."""

import os
import streamlit as st
import requests

# --- Config -------------------------------------------------------------------

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="V-RAG Assistant", page_icon="🧠", layout="wide")
st.title("🧠 V-RAG -- Video, PDF & Code Assistant")
st.write(
    "Extract insights, generate summaries, and chat with YouTube videos, "
    "PDF documents, or entire GitHub repositories."
)

# --- Session state init -------------------------------------------------------

for key, default in {
    "summary": None,
    "session_id": None,
    "chat_messages": [],
    "hf_api_token": None,
    "source_label": None,  # "video" | "pdf" | "repo"
    "file_count": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- Sidebar ------------------------------------------------------------------

with st.sidebar:
    st.header("🔑 API Keys")
    hf_api_token = st.text_input("Hugging Face Access Token", type="password")
    supadata_api_key = st.text_input(
        "Supadata API Key (for YouTube on cloud)",
        type="password",
        help="Free at supadata.ai -- 100 requests/month.",
    )
    github_token = st.text_input(
        "GitHub Token (for private repos)",
        type="password",
        help="Optional. Only needed for private repositories.",
    )

    st.markdown("---")

    if st.button("🗑️ Clear Session"):
        if st.session_state.session_id:
            try:
                requests.delete(
                    f"{BACKEND_URL}/api/session/{st.session_state.session_id}",
                    timeout=10,
                )
            except Exception:
                pass
        for key, default in {
            "summary": None, "session_id": None, "chat_messages": [],
            "hf_api_token": None, "source_label": None, "file_count": None,
        }.items():
            st.session_state[key] = default
        st.rerun()

# --- Source selection tabs -----------------------------------------------------

tab_video, tab_pdf, tab_repo = st.tabs(
    ["📺 YouTube Video", "📄 PDF Document", "🐙 GitHub Repo"]
)

# -- YouTube tab ---------------------------------------------------------------

with tab_video:
    video_input = st.text_input("YouTube Video URL or ID", key="video_url_input")
    process_video_btn = st.button("Process Video", key="process_video_btn")

    if process_video_btn:
        if not hf_api_token:
            st.error("Please provide a Hugging Face Access Token in the sidebar.")
        elif not video_input:
            st.error("Please enter a YouTube link or ID.")
        else:
            with st.spinner("Fetching transcript and building RAG pipeline..."):
                st.session_state.hf_api_token = hf_api_token
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/api/process",
                        json={
                            "video_url": video_input,
                            "hf_api_token": hf_api_token,
                            "supadata_api_key": supadata_api_key if supadata_api_key else None,
                        },
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.session_id = data["session_id"]
                        st.session_state.summary = data["summary"]
                        st.session_state.chat_messages = []
                        st.session_state.source_label = "video"
                        st.success("Video processed successfully!")
                    else:
                        error_detail = resp.json().get("detail", resp.text)
                        st.error(f"Error: {error_detail}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to the backend. Make sure the backend is running.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# -- PDF tab -------------------------------------------------------------------

with tab_pdf:
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        key="pdf_uploader",
        help="Max 10 MB. The text will be extracted and indexed for RAG.",
    )
    process_pdf_btn = st.button("Process PDF", key="process_pdf_btn")

    if process_pdf_btn:
        if not hf_api_token:
            st.error("Please provide a Hugging Face Access Token in the sidebar.")
        elif not uploaded_file:
            st.error("Please upload a PDF file.")
        else:
            with st.spinner("Extracting text and building RAG pipeline..."):
                st.session_state.hf_api_token = hf_api_token
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/api/process-pdf",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                        data={"hf_api_token": hf_api_token},
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.session_id = data["session_id"]
                        st.session_state.summary = data["summary"]
                        st.session_state.chat_messages = []
                        st.session_state.source_label = "pdf"
                        st.success(f"PDF '{uploaded_file.name}' processed successfully!")
                    else:
                        error_detail = resp.json().get("detail", resp.text)
                        st.error(f"Error: {error_detail}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to the backend. Make sure the backend is running.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# -- GitHub repo tab -----------------------------------------------------------

with tab_repo:
    repo_url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/owner/repo  or  owner/repo",
        key="repo_url_input",
    )
    process_repo_btn = st.button("Process Repository", key="process_repo_btn")

    if process_repo_btn:
        if not hf_api_token:
            st.error("Please provide a Hugging Face Access Token in the sidebar.")
        elif not repo_url:
            st.error("Please enter a GitHub repository URL.")
        else:
            with st.spinner("Downloading repo and building code index... (this may take a minute)"):
                st.session_state.hf_api_token = hf_api_token
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/api/process-repo",
                        json={
                            "repo_url": repo_url,
                            "hf_api_token": hf_api_token,
                            "github_token": github_token if github_token else None,
                        },
                        timeout=300,  # repos can take longer
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.session_id = data["session_id"]
                        st.session_state.summary = data["summary"]
                        st.session_state.file_count = data["file_count"]
                        st.session_state.chat_messages = []
                        st.session_state.source_label = "repo"
                        st.success(
                            f"Repository processed! Indexed {data['file_count']} files."
                        )
                    else:
                        error_detail = resp.json().get("detail", resp.text)
                        st.error(f"Error: {error_detail}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to the backend. Make sure the backend is running.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# --- Display summary ----------------------------------------------------------

if st.session_state.summary:
    label_map = {"video": "Video", "pdf": "Document", "repo": "Repository"}
    label = label_map.get(st.session_state.source_label, "Source")
    st.markdown("---")
    st.subheader(f"📋 {label} Summary")
    if st.session_state.file_count and st.session_state.source_label == "repo":
        st.caption(f"📁 {st.session_state.file_count} files indexed")
    st.markdown(st.session_state.summary)

# --- Quick actions for repo ---------------------------------------------------

if st.session_state.source_label == "repo" and st.session_state.session_id:
    st.markdown("---")
    st.subheader("⚡ Quick Actions")
    st.caption("Click a button to run a code analysis task:")

    col1, col2, col3, col4, col5 = st.columns(5)

    quick_action = None
    with col1:
        if st.button("🐛 Find Bugs", key="qa_bugs"):
            quick_action = (
                "Analyze the codebase and find potential bugs, security issues, "
                "and error-prone patterns. For each issue, cite the file path and "
                "line number, explain the problem, and suggest a fix."
            )
    with col2:
        if st.button("🧪 Generate Tests", key="qa_tests"):
            quick_action = (
                "Generate comprehensive unit tests for the main modules in this "
                "project. Use the appropriate testing framework for the language. "
                "Cite which file each test covers."
            )
    with col3:
        if st.button("📖 Documentation", key="qa_docs"):
            quick_action = (
                "Generate clear developer documentation for this project including: "
                "an overview, setup instructions, API reference for public functions "
                "and classes, and usage examples. Cite specific files."
            )
    with col4:
        if st.button("🔧 Refactor", key="qa_refactor"):
            quick_action = (
                "Review the codebase and suggest refactoring improvements for "
                "better readability, maintainability, and performance. Show the "
                "current code and proposed refactored version with file citations."
            )
    with col5:
        if st.button("🔍 Explain APIs", key="qa_api"):
            quick_action = (
                "Explain the API structure of this project. List all endpoints, "
                "public classes, and key functions with their signatures, parameters, "
                "return types, and purpose. Cite the file for each."
            )

    if quick_action:
        st.session_state.chat_messages.append({"role": "user", "content": quick_action})
        st.rerun()  # triggers the chat section below with the new message

# --- Chat ---------------------------------------------------------------------

st.markdown("---")
source_labels = {"video": "video", "pdf": "document", "repo": "repository"}
source = source_labels.get(st.session_state.source_label, "source")
st.subheader(f"💬 Chat with the {source}")

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Auto-send last unanswered message (for quick actions)
_needs_response = (
    st.session_state.chat_messages
    and st.session_state.chat_messages[-1]["role"] == "user"
    and st.session_state.session_id
)

if _needs_response:
    last_question = st.session_state.chat_messages[-1]["content"]
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                token = st.session_state.hf_api_token or hf_api_token
                resp = requests.post(
                    f"{BACKEND_URL}/api/chat",
                    json={
                        "session_id": st.session_state.session_id,
                        "question": last_question,
                        "hf_api_token": token,
                    },
                    timeout=120,
                )
                if resp.status_code == 200:
                    answer = resp.json()["answer"]
                    st.markdown(answer)
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": answer}
                    )
                elif resp.status_code == 404:
                    st.error("Session expired. Please process the source again.")
                else:
                    error_detail = resp.json().get("detail", resp.text)
                    st.error(f"Error: {error_detail}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the backend. Make sure the backend is running.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

if user_query := st.chat_input("Ask something..."):
    if not st.session_state.session_id:
        st.error("Please process a YouTube video, upload a PDF, or enter a GitHub repo first.")
    else:
        st.session_state.chat_messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    token = st.session_state.hf_api_token or hf_api_token
                    resp = requests.post(
                        f"{BACKEND_URL}/api/chat",
                        json={
                            "session_id": st.session_state.session_id,
                            "question": user_query,
                            "hf_api_token": token,
                        },
                        timeout=120,
                    )

                    if resp.status_code == 200:
                        answer = resp.json()["answer"]
                        st.markdown(answer)
                        st.session_state.chat_messages.append(
                            {"role": "assistant", "content": answer}
                        )
                    elif resp.status_code == 404:
                        st.error("Session expired. Please process the source again.")
                    else:
                        error_detail = resp.json().get("detail", resp.text)
                        st.error(f"Error: {error_detail}")

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to the backend. Make sure the backend is running.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
