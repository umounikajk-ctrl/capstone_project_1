import streamlit as st


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Zepto Support Assistant",
    page_icon="🛒",
    layout="centered"
)

st.title("Zepto Support Assistant")
st.caption("Ask questions about Zepto delivery, refunds, cancellations, gift cards, and support policies.")
st.divider()


# --------------------------------------------------
# Load assistant
# --------------------------------------------------

@st.cache_resource
def load_assistant():
    """
    Loads the assistant function once and reuses it across Streamlit reruns.
    This helps avoid reloading heavy RAG resources repeatedly.
    """
    from graph import ask_zepto
    return ask_zepto


try:
    ask_zepto = load_assistant()
except Exception as error:
    st.error(f"Failed to load assistant: {error}")
    st.stop()


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = None


# --------------------------------------------------
# Existing chat messages
# --------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            sources = message.get("sources", [])
            confidence = message.get("confidence")

            if sources or confidence is not None:
                with st.expander("Response details"):
                    if sources:
                        st.markdown("**Sources:**")
                        st.markdown(", ".join(sources))
                    else:
                        st.info("No source documents were used.")

                    if confidence is not None:
                        st.markdown(f"**Confidence:** {confidence}")


# --------------------------------------------------
# Chat input
# --------------------------------------------------

user_query = st.chat_input("Ask a Zepto policy question...")

if user_query:
    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching Zepto policy documents..."):
            try:
                result = ask_zepto(user_query)

                answer = result.get("answer", "No answer returned.")
                sources = result.get("sources", [])
                confidence = result.get("confidence", None)

                st.markdown(answer)

                with st.expander("Response details"):
                    if sources:
                        st.markdown("**Sources:**")
                        st.markdown(", ".join(sources))
                    else:
                        st.info("No source documents were used.")

                    if confidence is not None:
                        st.markdown(f"**Confidence:** {confidence}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "confidence": confidence
                })

                st.session_state.last_sources = sources
                st.session_state.last_confidence = confidence

            except Exception as error:
                st.error(f"Something went wrong: {error}")


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()
st.caption("Powered by ChromaDB, Sentence Transformers, LangGraph, and Streamlit.")