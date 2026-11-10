# AI Build frontend for RAG

import os
import time
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
import inngest
import requests

from main import inngest_client

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Inngest RAG Assistant",
    page_icon="⚡",
    layout="wide"
)

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
INNGEST_DEV_SERVER = os.getenv("INNGEST_DEV_SERVER_URL", "http://127.0.0.1:8288")


def poll_inngest_run(event_id: str, timeout: int = 45) -> dict:
    """
    Polls the Inngest Dev Server for the run triggered by an event_id
    until it finishes (Completed / Failed / Cancelled).
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(f"{INNGEST_DEV_SERVER}/v1/events/{event_id}/runs", timeout=5)
            if resp.status_code == 200:
                runs = resp.json().get("data", [])
                if runs:
                    latest_run = runs[0]
                    status = latest_run.get("status")
                    if status == "Completed":
                        return {
                            "status": "Completed",
                            "run_id": latest_run.get("run_id"),
                            "output": latest_run.get("output", {})
                        }
                    elif status in ["Failed", "Cancelled"]:
                        return {
                            "status": status,
                            "run_id": latest_run.get("run_id"),
                            "error": latest_run.get("output", {}).get("message", "Function execution failed.")
                        }
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)

    raise TimeoutError(f"Inngest run timed out after {timeout} seconds.")


def trigger_ingest_event(file_path: str, source_id: str) -> dict:
    """Sends 'rag/ingest_pdf' event to trigger rag_ingest_pdf function in main.py."""
    event = inngest.Event(
        name="rag/ingest_pdf",
        data={
            "pdf_path": str(file_path),
            "source_id": source_id
        }
    )
    event_ids = inngest_client.send_sync(event)
    if not event_ids:
        raise RuntimeError("Failed to send 'rag/ingest_pdf' event to Inngest.")
    
    event_id = event_ids[0]
    result = poll_inngest_run(event_id)
    return {
        "event_id": event_id,
        "run_id": result.get("run_id"),
        "ingested": result.get("output", {}).get("ingested", 0)
    }


def trigger_query_event(question: str, top_k: int = 5) -> dict:
    """Sends 'rag/query_pdf_ai' event to trigger rag_query_pdf_ai function in main.py."""
    event = inngest.Event(
        name="rag/query_pdf_ai",
        data={
            "question": question,
            "top_k": top_k
        }
    )
    event_ids = inngest_client.send_sync(event)
    if not event_ids:
        raise RuntimeError("Failed to send 'rag/query_pdf_ai' event to Inngest.")

    event_id = event_ids[0]
    result = poll_inngest_run(event_id)
    output = result.get("output", {})
    return {
        "event_id": event_id,
        "run_id": result.get("run_id"),
        "answer": output.get("answer", "No answer returned."),
        "sources": output.get("sources", []),
        "num_contexts": output.get("num_contexts", 0)
    }


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("📂 Document Ingestion")
    st.caption("Triggers the **`rag/ingest_pdf`** workflow in `main.py` via Inngest.")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("🚀 Ingest with Inngest", use_container_width=True, type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_ingested = 0

            for idx, uploaded_file in enumerate(uploaded_files):
                file_path = UPLOAD_DIR / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                status_text.text(f"Running Inngest workflow for: {uploaded_file.name}...")
                try:
                    res = trigger_ingest_event(str(file_path), uploaded_file.name)
                    total_ingested += res.get("ingested", 0)
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")

                progress_bar.progress((idx + 1) / len(uploaded_files))

            status_text.text("Ingestion workflow completed!")
            st.success(f"Successfully processed {len(uploaded_files)} file(s) into {total_ingested} chunks!")

    st.divider()
    st.header("⚙️ Query Configuration")
    top_k = st.slider("Top-k context chunks", min_value=1, max_value=10, value=5)

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("🔗 [Open Inngest Dev Dashboard](http://localhost:8288)")

# Main Chat Interface
st.title("⚡ Inngest RAG Assistant")
st.caption("Powered by `main.py` • Inngest Workflows • Qdrant • Gemini AI")

# Render previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 Sources & Inngest Run Info"):
                st.markdown(f"**Sources:** {', '.join(message['sources'])}")
                if message.get("run_id"):
                    st.caption(f"Inngest Run ID: `{message['run_id']}`")

# Chat input
if user_question := st.chat_input("Ask a question about your uploaded documents..."):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Dispatching 'rag/query_pdf_ai' event to Inngest and waiting for steps..."):
            try:
                res = trigger_query_event(user_question, top_k=top_k)
                answer = res["answer"]
                sources = res["sources"]
                run_id = res["run_id"]

                st.markdown(answer)

                if sources or run_id:
                    with st.expander("📚 Sources & Inngest Run Info"):
                        if sources:
                            st.markdown(f"**Sources:** {', '.join(sources)}")
                        if run_id:
                            st.caption(f"Inngest Run ID: `{run_id}`")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "run_id": run_id
                })
            except Exception as e:
                error_msg = f"❌ Execution failed: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
