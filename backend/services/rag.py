"""RAG engine — manages per-session FAISS vector stores, LLM chains, and chat history."""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.services.github_service import CodeFile

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace,
    HuggingFaceEndpointEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from operator import itemgetter


# ─── Session storage ──────────────────────────────────────────────────────────

SESSION_TTL_SECONDS = 3600  # 1 hour


@dataclass
class Session:
    """Holds all per-session RAG state."""

    session_id: str
    summary: str = ""
    chain: object = None  # RunnableWithMessageHistory
    chat_history: InMemoryChatMessageHistory = field(
        default_factory=InMemoryChatMessageHistory
    )
    created_at: float = field(default_factory=time.time)


# Global session store
_sessions: dict[str, Session] = {}


def _cleanup_expired_sessions() -> None:
    """Remove sessions older than SESSION_TTL_SECONDS."""
    now = time.time()
    expired = [
        sid
        for sid, sess in _sessions.items()
        if now - sess.created_at > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        del _sessions[sid]


def get_session(session_id: str) -> Session | None:
    """Retrieve an existing session by ID."""
    _cleanup_expired_sessions()
    return _sessions.get(session_id)


def delete_session(session_id: str) -> bool:
    """Delete a session. Returns True if it existed."""
    return _sessions.pop(session_id, None) is not None


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _format_docs(retrieved_docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def _format_code_docs(retrieved_docs: list[Document]) -> str:
    """Format retrieved code chunks with file path citations."""
    parts: list[str] = []
    for doc in retrieved_docs:
        file_path = doc.metadata.get("file_path", "unknown")
        language = doc.metadata.get("language", "")
        start_line = doc.metadata.get("start_line", "")
        end_line = doc.metadata.get("end_line", "")

        line_info = ""
        if start_line and end_line:
            line_info = f" (lines {start_line}-{end_line})"
        elif start_line:
            line_info = f" (line {start_line}+)"

        header = f"### File: {file_path}{line_info}"
        code_block = f"```{language}\n{doc.page_content}\n```"
        parts.append(f"{header}\n{code_block}")
    return "\n\n".join(parts)


# ─── Source-type prompts ──────────────────────────────────────────────────────

_SYSTEM_PROMPTS = {
    "video": (
        "You are an AI assistant analyzing a video transcript. "
        "Answer using ONLY the context below.\n\nContext:\n{context}"
    ),
    "pdf": (
        "You are an AI assistant analyzing a PDF document. "
        "Answer using ONLY the context below.\n\nContext:\n{context}"
    ),
    "repo": (
        "You are an expert code analyst reviewing a software repository. "
        "You can find bugs, suggest fixes, explain APIs, generate tests, "
        "refactor code, and create documentation.\n\n"
        "IMPORTANT RULES:\n"
        "- ALWAYS cite the file path when referencing code (e.g., `src/utils.py`).\n"
        "- When showing code, include the file path as a heading.\n"
        "- When finding bugs, specify the file and line number if possible.\n"
        "- When generating tests or refactored code, format as complete code blocks.\n"
        "- Base your answers ONLY on the code context provided below.\n\n"
        "Code context:\n{context}"
    ),
}

_SUMMARY_PROMPTS = {
    "video": (
        "Provide a systematic, bulleted summary highlighting the core "
        "concepts from this video context.\n\nContext:\n{context}"
    ),
    "pdf": (
        "Provide a systematic, bulleted summary highlighting the key points, "
        "arguments, and conclusions from this document.\n\nContext:\n{context}"
    ),
    "repo": (
        "You are analyzing a software repository. Provide a structured project overview "
        "covering:\n"
        "- **Project purpose**: What does this project do?\n"
        "- **Tech stack**: Languages, frameworks, and key libraries used\n"
        "- **Project structure**: Main directories and their purposes\n"
        "- **Key files**: The most important files and what they contain\n"
        "- **Architecture**: How the components connect\n\n"
        "Repository contents:\n{context}"
    ),
}

_SUMMARY_QUESTIONS = {
    "video": "Summarize the video contents structurally.",
    "pdf": "Summarize the document contents structurally.",
    "repo": "Analyze this repository and provide a structured project overview.",
}


# ─── Core pipeline ────────────────────────────────────────────────────────────


def process_text(
    text: str, hf_api_token: str, source_type: str = "video"
) -> tuple[str, str]:
    """
    Build a RAG pipeline from raw text (video transcript or PDF content).

    Args:
        text: The full source text.
        hf_api_token: HuggingFace API token.
        source_type: "video" or "pdf" — adapts prompts accordingly.

    Returns:
        (session_id, summary)
    """
    _cleanup_expired_sessions()
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_token

    if source_type not in _SYSTEM_PROMPTS:
        source_type = "video"  # safe fallback

    # 1. Chunk the text
    doc = Document(page_content=text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents([doc])

    if not chunks:
        raise ValueError("Text produced no chunks after splitting.")

    # 2. Embeddings + FAISS
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        huggingfacehub_api_token=hf_api_token,
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    # 3. LLM
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        task="text-generation",
        max_new_tokens=512,
        huggingfacehub_api_token=hf_api_token,
    )
    chat_model = ChatHuggingFace(llm=llm)

    # 4. Chat chain with history
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPTS[source_type]),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    session = Session(session_id=str(uuid.uuid4()))

    context_and_question = RunnablePassthrough.assign(
        context=itemgetter("question") | retriever | RunnableLambda(_format_docs)
    )

    base_chain = context_and_question | prompt | chat_model | StrOutputParser()

    session.chain = RunnableWithMessageHistory(
        base_chain,
        lambda _sid: session.chat_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    # 5. Generate summary
    summary_text = text[:3000]
    summary_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SUMMARY_PROMPTS[source_type]),
            ("human", "{question}"),
        ]
    )
    summary_chain = summary_prompt | chat_model | StrOutputParser()
    session.summary = summary_chain.invoke(
        {"context": summary_text, "question": _SUMMARY_QUESTIONS[source_type]}
    )

    # Store session
    _sessions[session.session_id] = session

    return session.session_id, session.summary


def process_video(transcript_text: str, hf_api_token: str) -> tuple[str, str]:
    """Backward-compatible wrapper — processes a video transcript."""
    return process_text(transcript_text, hf_api_token, source_type="video")


def process_repo(
    files: list[CodeFile], hf_api_token: str
) -> tuple[str, str, int]:
    """
    Build a RAG pipeline from repository code files.

    Each code file is chunked with file-path metadata so the LLM can cite
    specific files and line numbers in its responses.

    Args:
        files: List of CodeFile objects from github_service.
        hf_api_token: HuggingFace API token.

    Returns:
        (session_id, summary, file_count)
    """
    _cleanup_expired_sessions()
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_token

    # 1. Create documents with file metadata
    all_chunks: list[Document] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\nclass ", "\ndef ", "\n\n", "\n", " "],
    )

    for code_file in files:
        doc = Document(
            page_content=code_file.content,
            metadata={
                "file_path": code_file.path,
                "language": code_file.language,
            },
        )
        file_chunks = splitter.split_documents([doc])

        # Add line number estimates to each chunk
        lines = code_file.content.split("\n")
        for chunk in file_chunks:
            # Estimate start line by finding the chunk text in the original
            chunk_start = code_file.content.find(chunk.page_content[:80])
            if chunk_start >= 0:
                start_line = code_file.content[:chunk_start].count("\n") + 1
                end_line = start_line + chunk.page_content.count("\n")
                chunk.metadata["start_line"] = start_line
                chunk.metadata["end_line"] = end_line

        all_chunks.extend(file_chunks)

    if not all_chunks:
        raise ValueError("No code chunks produced after splitting.")

    # 2. Embeddings + FAISS
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        huggingfacehub_api_token=hf_api_token,
    )
    vector_store = FAISS.from_documents(all_chunks, embeddings)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 6}
    )

    # 3. LLM
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        task="text-generation",
        max_new_tokens=1024,  # larger for code output
        huggingfacehub_api_token=hf_api_token,
    )
    chat_model = ChatHuggingFace(llm=llm)

    # 4. Chat chain with history — uses code-aware doc formatter
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPTS["repo"]),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    session = Session(session_id=str(uuid.uuid4()))

    context_and_question = RunnablePassthrough.assign(
        context=itemgetter("question") | retriever | RunnableLambda(_format_code_docs)
    )

    base_chain = context_and_question | prompt | chat_model | StrOutputParser()

    session.chain = RunnableWithMessageHistory(
        base_chain,
        lambda _sid: session.chat_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    # 5. Generate project overview summary
    #    Build a file tree + sample content for the summary
    file_tree = "\n".join(f"  - {f.path} ({f.language})" for f in files)
    sample_content = "\n\n".join(
        f"### {f.path}\n```{f.language}\n{f.content[:500]}\n```"
        for f in files[:15]  # first 15 files for summary
    )
    summary_context = (
        f"## File tree ({len(files)} files):\n{file_tree}\n\n"
        f"## Sample file contents:\n{sample_content}"
    )[:4000]  # cap context

    summary_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SUMMARY_PROMPTS["repo"]),
            ("human", "{question}"),
        ]
    )
    summary_chain = summary_prompt | chat_model | StrOutputParser()
    session.summary = summary_chain.invoke(
        {"context": summary_context, "question": _SUMMARY_QUESTIONS["repo"]}
    )

    _sessions[session.session_id] = session

    return session.session_id, session.summary, len(files)


def chat(session_id: str, question: str, hf_api_token: str) -> str:
    """Send a question to an existing session's RAG chain. Returns the answer."""
    session = get_session(session_id)
    if session is None:
        raise ValueError(f"Session '{session_id}' not found or expired.")
    if session.chain is None:
        raise ValueError("Session exists but has no RAG chain.")

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_token
    config = {"configurable": {"session_id": session_id}}
    return session.chain.invoke({"question": question}, config=config)
