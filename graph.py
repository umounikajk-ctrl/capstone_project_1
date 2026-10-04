import os
import json
import time
from typing import TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from openai import OpenAI
from ingest import collection, embed_text
from prompts import PROMPT_TEMPLATE


load_dotenv()


#--------------------------------------------------
# MOCK LLM
# ---------------------------------------------

# Default = mock mode
# MOCK_LLM=1 -> True
# MOCK_LLM=0 -> False

MOCK_LLM = os.getenv("MOCK_LLM", "1") == "1"


client = None


if not MOCK_LLM:

    groq_key = os.getenv("GROQ_API_KEY")

    if not groq_key:
        raise ValueError(
            "GROQ_API_KEY is required when MOCK_LLM=0"
        )

    client = OpenAI(
        api_key=groq_key,
        base_url="https://api.groq.com/openai/v1"
    )


# -----------------------------------------
# LLM CALL
# -----------------------------------------

def call_llm(
    prompt: str,
    system: str = "You are a helpful Zepto support assistant.") -> str:

    """Simple helper to call the LLM."""

    response = client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[
            {"role": "system","content": system},
            {"role": "user","content": prompt}
            ],
        temperature=0.2
        )
    return response.choices[0].message.content


# -------------------------------------------
# PYDANTIC RESPONSE SCHEMA
#-------------------------------------------
class SupportResponse(BaseModel):

    answer: str

    sources: list[str]

    confidence: float = Field(ge=0.0,le=1.0)


# ---------------------------------------------
# STATE
# ---------------------------------------------

class SimpleState(TypedDict):
    query: str
    intent: str
    answer: str
    sources: list[str]
    confidence: float


# ----------------------------------------------
# VALIDATE REAL LLM RESPONSE
#-----------------------------------------------

def generate_valid_response(prompt: str,system: str,expected_sources: list[str]):

    """
    Calls the real LLM and validates its JSON output
    using the SupportResponse Pydantic model.

    Maximum attempts = 3
    """

    corrective_instruction = ""

    for attempt in range(3):

        try:

            # Add corrective instruction on retry
            current_prompt = prompt + corrective_instruction

            raw_response = call_llm(current_prompt,system=system)

            # Remove markdown JSON fences if LLM adds them
            raw_response = raw_response.strip()

            if raw_response.startswith("```json"):
                raw_response = raw_response[7:]

            elif raw_response.startswith("```"):
                raw_response = raw_response[3:]

            if raw_response.endswith("```"):
                raw_response = raw_response[:-3]

            raw_response = raw_response.strip()

            # Convert JSON string -> Python dictionary
            data = json.loads(raw_response)

            # Pydantic validation
            response = SupportResponse.model_validate(data)


            # Use actual retrieved sources
            # instead of trusting LLM-generated source IDs
            response = SupportResponse(
                answer=response.answer,
                sources=expected_sources,
                confidence=response.confidence
            )

            return response.model_dump()

        except Exception as e:

            print(
                f"LLM response validation failed "
                f"(attempt {attempt + 1}/3): {e}")

            if attempt < 2:
                corrective_instruction = """

                    IMPORTANT CORRECTION:

                    Your previous response was invalid.

                    Return ONLY valid JSON.

                    The JSON must contain exactly these fields:

                    {
                       "answer": "string",
                       "sources": ["string"],
                       "confidence": 0.0
                    }

                    Rules:
                    - answer must be a string
                    - sources must be a list of strings
                    - confidence must be a number between 0 and 1
                    - Do not use markdown
                    - Do not add explanations outside the JSON
                    """

                time.sleep(2 ** attempt)

    # All 3 attempts failed
    error_response = SupportResponse(

        answer=(
            "ERROR: Unable to produce a valid "
            "structured response after 3 attempts."),

        sources=expected_sources,

        confidence=0.0
    )

    return error_response.model_dump()
#------------------------------------------
# NODE 1: CLASSIFY INTENT
#-------------------------------------------
def classify_intent(state: SimpleState):

    query = state["query"].lower()

    keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "cancel",
        "gift card",
        "support hours"
    ]

    # MOCK MODE
    if MOCK_LLM:

        if any(keyword in query for keyword in keywords):

            return {
                "intent": "policy_question"
            }

        else:

            return {
                "intent": "general_question"
            }

    # REAL LLM MODE
    for attempt in range(3):

        try:
            intent = call_llm(
                f"""
                Classify the following question into exactly one category:

                policy_question
                general_question

                Question:
                {query}

                Return ONLY one of these two values:
                policy_question
                general_question
                """,

                system=(
                    "You classify Zepto support questions. "
                    "Return exactly one category.")).strip().lower()

            # Validate intent
            if intent not in {"policy_question","general_question"}:
                raise ValueError("Invalid intent returned by LLM")

            return {
                "intent": intent
            }

        except Exception as e:

            print(
                f"Intent classification failed "
                f"(attempt {attempt + 1}/3): {e}"
            )

            if attempt < 2:
                time.sleep(2 ** attempt)

    # Safe fallback

    return {
        "intent": "general_question"
    }
#-------------------------------------------
# ROUTER
#------------------------------------------
def router(state: SimpleState):

    if state["intent"] == "policy_question":

        return "retrieve"

    return "general"
#--------------------------------------------
# NODE 2: RETRIEVE AND ANSWER
#--------------------------------------------
def retrieve_and_answer(state: SimpleState):

    query = state["query"]
    # Create query embedding
    query_emb = embed_text(query)

    # Retrieve top 3 documents
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=3
    )

    documents = results["documents"][0]

    source_ids = results["ids"][0]

    # Most similar document
    top_chunk = documents[0]

    # Mock response snippet
    top_chunk_snippet = (
        top_chunk[:200]
        .replace("\n", " ")
        .strip()
    )

    # Combine retrieved documents
    retrieved_context = "\n\n".join(documents)

    # Structured prompt
    prompt = PROMPT_TEMPLATE.format(
        context=retrieved_context,
        question=query
    )

    # MOCK MODE
    if MOCK_LLM:

        response = SupportResponse(

            answer=(
                f"Based on the retrieved context: "
                f"{top_chunk_snippet}"
            ),

            sources=source_ids,

            confidence=1.0
        )

        return response.model_dump()

    # REAL LLM MODE
    response = generate_valid_response(

        prompt=prompt,

        system=(
            "You are a Zepto policy assistant. "
            "Return only valid JSON."
        ),

        expected_sources=source_ids
    )

    return response

#------------------------------------------
# NODE 3: DIRECT ANSWER
#------------------------------------------
def direct_answer(state: SimpleState):

    query = state["query"]

    # Prompt for real LLM
    prompt = f"""
              ROLE:
              You are a Zepto support assistant.
              TASK:
              Answer the user's question directly.
              QUESTION:
              {query}
              FORMAT:
              Return only valid JSON.
              Required format:
             {{
                "answer": "string",
                "sources": [],
                "confidence": 0.0
             }}
             LENGTH:
             Keep the answer within 2-3 sentences.
             NEGATIVE CONSTRAINT:
             Do not invent Zepto policies.
             """
    
    # MOCK MODE
    if MOCK_LLM:

        response = SupportResponse(

            answer=(
                "I can only answer questions about "
                "Zepto policies right now."
            ),

            sources=[],

            confidence=1.0
        )

        return response.model_dump()

    # REAL LLM MODE
    response = generate_valid_response(prompt=prompt,
        system=(
            "You are a Zepto support assistant. "
            "Return only valid JSON."
        ),

        expected_sources=[]
    )

    return response

#----------------------------------------
# BUILD LANGGRAPH
#-------------------------------------------
graph = StateGraph(SimpleState)

# Add nodes
graph.add_node("classify_intent",classify_intent)
graph.add_node("retrieve_and_answer",retrieve_and_answer)
graph.add_node("direct_answer",direct_answer)

# START -> classify_intent
graph.add_edge(START,"classify_intent")

# Conditional routing
graph.add_conditional_edges("classify_intent",router,

    {
        "retrieve": "retrieve_and_answer",
        "general": "direct_answer"
    }
)


# Answer nodes -> END
graph.add_edge("retrieve_and_answer",END)
graph.add_edge("direct_answer",END)

# Compile graph
graph_app = graph.compile()

# FUNCTION USED BY FASTAPI
def ask_zepto(query: str):

    result = graph_app.invoke(

        {
            "query": query,
            "intent": "",
            "answer": "",
            "sources": [],
            "confidence": 0.0
        }

    )

    return result

# TEST
if __name__ == "__main__":

    print(ask_zepto("what is the capital of India"))