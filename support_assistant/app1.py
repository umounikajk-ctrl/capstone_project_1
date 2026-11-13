import os
import requests
import streamlit as st


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Zepto Support Assistant",
    page_icon="Z",
    layout="centered"
)

st.title("Zepto Support Assistant")
st.caption("Ask questions about Zepto policies using the FastAPI backend.")
st.divider()


# --------------------------------------------------
# FastAPI configuration
# --------------------------------------------------

API_BASE_URL = os.getenv(
    "FASTAPI_BASE_URL",
    "http://127.0.0.1:7860"
)

ASK_ENDPOINT = f"{API_BASE_URL}/ask"
HEALTH_ENDPOINT = f"{API_BASE_URL}/"


@st.cache_resource
def get_http_session():
    """
    Creates one reusable HTTP session for Streamlit reruns.
    """
    return requests.Session()


session = get_http_session()


# --------------------------------------------------
# Check FastAPI backend
# --------------------------------------------------

try:
    response = session.get(HEALTH_ENDPOINT, timeout=5)

    if response.status_code != 200:
        st.error("FastAPI backend is running but returned an unexpected response.")
        st.stop()

except requests.exceptions.RequestException:
    st.error(
        "FastAPI backend is not reachable. "
        "Start it first using: uvicorn main:app --reload --port 7860"
    )
    st.stop()


st.success("Connected to FastAPI backend.")


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# Display previous chat messages
# --------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            sources = message.get("sources", [])
            confidence = message.get("confidence")

            with st.expander("Response details"):
                if sources:
                    st.markdown("**Sources:**")
                    st.markdown(", ".join(sources))
                else:
                    st.info("No source documents were returned.")

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
        with st.spinner("Sending question to FastAPI backend..."):
            try:
                api_response = session.post(
                    ASK_ENDPOINT,
                    json={
                        "query": user_query
                    },
                    timeout=30
                )

                api_response.raise_for_status()
                result = api_response.json()

                answer = result.get("answer", "No answer returned.")
                sources = result.get("sources", [])
                confidence = result.get("confidence")

                st.markdown(answer)

                with st.expander("Response details"):
                    if sources:
                        st.markdown("**Sources:**")
                        st.markdown(", ".join(sources))
                    else:
                        st.info("No source documents were returned.")

                    if confidence is not None:
                        st.markdown(f"**Confidence:** {confidence}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "confidence": confidence
                })

            except requests.exceptions.RequestException as error:
                st.error(f"FastAPI request failed: {error}")

            except ValueError:
                st.error("FastAPI returned an invalid JSON response.")


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()
st.caption(f"Backend API: {API_BASE_URL}")
