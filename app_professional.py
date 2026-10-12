from backend_professional import (
    chatbot,
    get_all_threads,
    ingest_rag_document
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage
)

from langgraph.types import Command

import streamlit as st
import uuid
import tempfile
import os


# ========================= Utility functions =========================

def generate_thread_id():
    return str(uuid.uuid4())


def short_thread_label(thread_id: str) -> str:
    return f"Chat {thread_id[:8]}"


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []
    st.session_state["pending_hitl"] = None
    add_thread(st.session_state["thread_id"])


def load_conversation(thread_id):
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )
    return state.values.get("messages", [])


# ========================= HITL helper functions =========================

def get_pending_interrupt(thread_id):
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:
        state_snapshot = chatbot.get_state(config)
        direct_interrupts = getattr(state_snapshot, "interrupts", ()) or ()
        if direct_interrupts:
            return direct_interrupts[0]

        tasks = getattr(state_snapshot, "tasks", ()) or ()
        for task in tasks:
            task_interrupts = getattr(task, "interrupts", ()) or ()
            if task_interrupts:
                return task_interrupts[0]

    except Exception:
        return None

    return None


def save_pending_interrupt(thread_id, interrupt_object):
    st.session_state["pending_hitl"] = {
        "thread_id": thread_id,
        "prompt": str(interrupt_object.value)
    }


def sync_pending_interrupt(thread_id):
    pending_interrupt = get_pending_interrupt(thread_id)

    if pending_interrupt is not None:
        save_pending_interrupt(thread_id, pending_interrupt)
    else:
        current_pending = st.session_state.get("pending_hitl")
        if current_pending is not None and current_pending.get("thread_id") == thread_id:
            st.session_state["pending_hitl"] = None


def resume_hitl_execution(decision):
    pending_hitl = st.session_state.get("pending_hitl")

    if not pending_hitl:
        st.warning("There is no pending action to approve or reject.")
        return

    interrupted_thread_id = pending_hitl["thread_id"]

    resume_config = {
        "configurable": {"thread_id": interrupted_thread_id},
        "metadata": {"thread_id": interrupted_thread_id},
        "run_name": "hitl_resume_trace",
    }

    try:
        with st.chat_message("assistant"):
            status_holder = {
                "box": st.status("🔄 Resuming the requested action...", expanded=True)
            }

            def resumed_ai_only_stream():
                for message_chunk, metadata in chatbot.stream(
                    Command(resume=decision),
                    config=resume_config,
                    stream_mode="messages",
                ):
                    if isinstance(message_chunk, ToolMessage):
                        tool_name = getattr(message_chunk, "name", "tool")
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                    if isinstance(message_chunk, AIMessage) and message_chunk.content:
                        yield message_chunk.content

            resumed_ai_message = st.write_stream(resumed_ai_only_stream())
            next_interrupt = get_pending_interrupt(interrupted_thread_id)

            if next_interrupt is not None:
                save_pending_interrupt(interrupted_thread_id, next_interrupt)
                status_holder["box"].update(
                    label="⚠️ Another approval is required",
                    state="complete",
                    expanded=False
                )
            else:
                st.session_state["pending_hitl"] = None
                status_holder["box"].update(
                    label="✅ Action completed",
                    state="complete",
                    expanded=False
                )

        if resumed_ai_message:
            st.session_state["message_history"].append({
                "role": "assistant",
                "content": resumed_ai_message
            })

        st.rerun()

    except Exception as error:
        st.error(f"Could not resume the requested action: {error}")


# ========================= Styling =========================

def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(96, 165, 250, 0.16), transparent 35%),
                radial-gradient(circle at top right, rgba(129, 140, 248, 0.14), transparent 30%),
                linear-gradient(180deg, #0b1020 0%, #111827 40%, #0f172a 100%);
            color: #e5e7eb;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.15);
        }

        h1, h2, h3 {
            color: #f8fafc !important;
            letter-spacing: -0.02em;
        }

        .hero-card {
            padding: 1.4rem 1.4rem 1.1rem 1.4rem;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(30,41,59,0.80), rgba(15,23,42,0.92));
            border: 1px solid rgba(148, 163, 184, 0.18);
            box-shadow: 0 10px 30px rgba(2, 6, 23, 0.35);
            margin-bottom: 1rem;
        }

        .hero-title {
            font-size: 2rem;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 0.25rem;
        }

        .hero-subtitle {
            color: #cbd5e1;
            font-size: 1rem;
            line-height: 1.6;
        }

        .mini-card {
            padding: 0.95rem 1rem;
            border-radius: 18px;
            background: rgba(15, 23, 42, 0.70);
            border: 1px solid rgba(148, 163, 184, 0.12);
            margin-bottom: 0.8rem;
        }

        .mini-label {
            color: #94a3b8;
            font-size: 0.82rem;
            margin-bottom: 0.15rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .mini-value {
            color: #f8fafc;
            font-size: 1rem;
            font-weight: 700;
        }

        .feature-chip {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            margin: 0.25rem 0.35rem 0.1rem 0;
            border-radius: 999px;
            background: rgba(59, 130, 246, 0.14);
            color: #bfdbfe;
            border: 1px solid rgba(96, 165, 250, 0.18);
            font-size: 0.84rem;
            font-weight: 600;
        }

        [data-testid="stChatMessage"] {
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid rgba(148, 163, 184, 0.10);
            border-radius: 18px;
            padding: 0.45rem 0.25rem;
            backdrop-filter: blur(8px);
        }

        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] li,
        [data-testid="stChatMessage"] span {
            color: #e5e7eb !important;
            font-size: 0.98rem;
            line-height: 1.7;
        }

        .stButton > button {
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.20);
            background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
            color: #f8fafc;
            font-weight: 600;
        }

        .stButton > button:hover {
            border-color: rgba(96, 165, 250, 0.45);
            color: #ffffff;
        }

        [data-testid="stFileUploaderDropzone"],
        [data-testid="stChatInput"] {
            border-radius: 16px !important;
        }

        .empty-state {
            padding: 1.2rem 1.2rem;
            border-radius: 20px;
            background: rgba(15, 23, 42, 0.62);
            border: 1px dashed rgba(148, 163, 184, 0.20);
            color: #cbd5e1;
            margin-top: 0.5rem;
            margin-bottom: 1rem;
        }

        .approval-box {
            padding: 1rem 1rem 0.3rem 1rem;
            border-radius: 18px;
            background: rgba(120, 53, 15, 0.18);
            border: 1px solid rgba(251, 191, 36, 0.28);
            color: #fde68a;
            margin: 0.7rem 0 1rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Agentic Chatbot</div>
            <div class="hero-subtitle">
                A polished LangGraph + LangChain assistant with web search, weather, stock tools,
                PDF-based RAG, and human-in-the-loop approval.
            </div>
            <div style="margin-top:0.8rem;">
                <span class="feature-chip">LangGraph Memory</span>
                <span class="feature-chip">PDF RAG</span>
                <span class="feature-chip">Tool Use</span>
                <span class="feature-chip">Human Approval</span>
                <span class="feature-chip">Streaming UX</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ========================= Page configuration =========================

st.set_page_config(
    page_title="Agentic Chatbot",
    page_icon="🤖",
    layout="wide"
)

inject_custom_css()

# ========================= Session state initialization =========================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

if "pending_hitl" not in st.session_state:
    st.session_state["pending_hitl"] = None

add_thread(st.session_state["thread_id"])
sync_pending_interrupt(st.session_state["thread_id"])


# ========================= Sidebar =========================

with st.sidebar:
    st.markdown("## Conversations")
    st.caption("Manage your chat threads and continue past discussions.")

    if st.button("✨ New Chat", use_container_width=True, type="primary"):
        reset_chat()
        st.rerun()

    st.markdown("---")

    st.markdown(
        f"""
        <div class="mini-card">
            <div class="mini-label">Active Thread</div>
            <div class="mini-value">{short_thread_label(st.session_state['thread_id'])}</div>
        </div>
        <div class="mini-card">
            <div class="mini-label">Saved Conversations</div>
            <div class="mini-value">{len(st.session_state['chat_threads'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Recent threads")
    for thread_id in st.session_state["chat_threads"][::-1]:
        is_active = thread_id == st.session_state["thread_id"]
        label = ("🟢 " if is_active else "⚪ ") + short_thread_label(thread_id)

        if st.button(label, key=thread_id, use_container_width=True):
            st.session_state["thread_id"] = thread_id
            messages = load_conversation(thread_id)
            temp_messages = []

            for message in messages:
                if isinstance(message, HumanMessage):
                    role = "user"
                elif isinstance(message, AIMessage):
                    role = "assistant"
                else:
                    continue

                temp_messages.append({
                    "role": role,
                    "content": message.content
                })

            st.session_state["message_history"] = temp_messages
            sync_pending_interrupt(thread_id)
            st.rerun()

    st.markdown("---")
    st.markdown(
        """
        <div class="mini-card">
            <div class="mini-label">Capabilities</div>
            <div style="color:#cbd5e1; line-height:1.7;">
                • Document Q&A<br>
                • Web search<br>
                • Weather<br>
                • Stock data<br>
                • Particle physics Q&A<br>
                • Human approval flow
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ========================= Main content =========================

render_header()

col1, col2, col3 = st.columns(3)
col1.markdown(
    """
    <div class="mini-card">
        <div class="mini-label">Model Layer</div>
        <div class="mini-value">OpenAI-compatible LLM</div>
    </div>
    """,
    unsafe_allow_html=True,
)
col2.markdown(
    """
    <div class="mini-card">
        <div class="mini-label">Knowledge Layer</div>
        <div class="mini-value">PDF RAG + FAISS</div>
    </div>
    """,
    unsafe_allow_html=True,
)
col3.markdown(
    """
    <div class="mini-card">
        <div class="mini-label">Interaction Layer</div>
        <div class="mini-value">Streaming + HITL</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Display all messages
if not st.session_state["message_history"]:
    st.markdown(
        """
        <div class="empty-state">
            <strong>Welcome.</strong> Start a conversation below, or upload a PDF directly from the chat input to activate document-based question answering.
        </div>
        """,
        unsafe_allow_html=True,
    )

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ========================= HITL approval interface =========================

pending_hitl = st.session_state.get("pending_hitl")
current_thread_has_pending_hitl = (
    pending_hitl is not None
    and pending_hitl.get("thread_id") == st.session_state["thread_id"]
)

if current_thread_has_pending_hitl:
    st.markdown(
        f"""
        <div class="approval-box">
            <strong>🧑 Human approval required</strong><br><br>
            {pending_hitl['prompt']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    approve_column, reject_column = st.columns(2)

    with approve_column:
        if st.button(
            "✅ Approve Purchase",
            key=f"approve_{st.session_state['thread_id']}",
            type="primary",
            use_container_width=True
        ):
            resume_hitl_execution("yes")

    with reject_column:
        if st.button(
            "❌ Reject Purchase",
            key=f"reject_{st.session_state['thread_id']}",
            use_container_width=True
        ):
            resume_hitl_execution("no")


# ========================= Chat input with PDF upload =========================

submission = st.chat_input(
    "Ask anything... or upload a PDF for RAG",
    accept_file=True,
    file_type=["pdf"],
    disabled=current_thread_has_pending_hitl
)

user_input = None

if submission:
    user_input = submission.text
    uploaded_files = submission.files

    if uploaded_files:
        uploaded_pdf = uploaded_files[0]
        temporary_file_path = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temporary_file:
                temporary_file.write(uploaded_pdf.getvalue())
                temporary_file_path = temporary_file.name

            with st.spinner(f"Processing {uploaded_pdf.name}..."):
                ingest_rag_document(temporary_file_path)

            st.toast(f"{uploaded_pdf.name} processed successfully.", icon="✅")

        except Exception as error:
            st.error(f"PDF processing failed: {error}")

        finally:
            if temporary_file_path and os.path.exists(temporary_file_path):
                os.remove(temporary_file_path)


# ========================= Chat execution =========================

if user_input:
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_trace",
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")

                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …",
                            expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

            pending_interrupt = get_pending_interrupt(st.session_state["thread_id"])

            if pending_interrupt is not None:
                save_pending_interrupt(st.session_state["thread_id"], pending_interrupt)
                yield (
                    "\n\n⚠️ This stock purchase requires your approval. "
                    "Use the Approve Purchase or Reject Purchase button below."
                )

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            if get_pending_interrupt(st.session_state["thread_id"]) is not None:
                status_holder["box"].update(
                    label="⏸️ Waiting for human approval",
                    state="complete",
                    expanded=False
                )
            else:
                status_holder["box"].update(
                    label="✅ Tool finished",
                    state="complete",
                    expanded=False
                )

    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })

    if (
        st.session_state.get("pending_hitl") is not None
        and st.session_state["pending_hitl"].get("thread_id") == st.session_state["thread_id"]
    ):
        st.rerun()
