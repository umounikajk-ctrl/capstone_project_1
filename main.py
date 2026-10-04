from fastapi import FastAPI
from pydantic import BaseModel, Field

from graph import ask_zepto


#---------------------------------
# REQUEST MODEL
# ---------------------------------

class ChatRequest(BaseModel):
    """
    Shape of the POST /ask request body.

    Example:
        {"query": "What is the delivery policy?"}
    """

    query: str = Field(min_length=1)


# ----------------------------------------
# RESPONSE MODEL
# ---------------------------------------

class ChatResponse(BaseModel):

    answer: str

    sources: list[str]

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )


#---------------------------------------
# FASTAPI APPLICATION
#--------------------------------------

app = FastAPI(
    title="Zepto Support Assistant",
    version="1.0"
)


@app.get("/")
def home():

    return {
        "status": "running",
        "message": "Zepto support assistant is live!"
    }


# -------------------------------------
# ASK ENDPOINT
# -------------------------------------

@app.post(
    "/ask",
    response_model=ChatResponse
)
def ask(request: ChatRequest):

    result = ask_zepto(request.query)

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )