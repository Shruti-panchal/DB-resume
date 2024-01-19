from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2AuthorizationCodeBearer
from fastapi_limiter import FastAPILimiter
from pydantic import BaseModel

import spacy
from nltk import sent_tokenize, pos_tag, word_tokenize
from pycontractions import Contractions
import re
import language_tool_python
from textblob import TextBlob

# Create a FastAPI app
app = FastAPI()

# OAuth2 for token-based authentication
# oauth2_scheme = OAuth2AuthorizationCodeBearer(tokenUrl="token", authorizationUrl="token")

# Initialize rate limiting
# @app.on_event("startup")
# async def startup_event():
#     await FastAPILimiter.init()

# @app.get("/")
# async def read_root():
#     return {"message": "Welcome to the API"}


# Define a Pydantic model for the input data
class TextRequest(BaseModel):
    text: str

# Endpoint for processing text with security measures
@app.post("/process-text/", response_model=dict)
async def process_text(request: TextRequest):
    # , token: str = Depends(oauth2_scheme)
    # Your existing text processing logic goes here
    nlp = spacy.load("en_core_web_sm")

    tool = TextBlob(request.text)
    print("Original Text: ", request.text)

    print("#############################################################################################processing")

    tool2 = language_tool_python.LanguageTool('en-US')

    cont = Contractions(api_key="glove-twitter-100")
    cont.load_models()

    ex_text = list(cont.expand_texts([request.text]))[0]
    print("Expanded text : ", ex_text)

    cleaned = re.sub('[^A-Za-z0-9.,? ]', '', ex_text)
    print("text after cleaning:: ", cleaned)

    tokens = word_tokenize(cleaned)
    print("Token: ", tokens)

    tags = pos_tag(tokens)
    print("POS Tags: ", tags)

    gr1 = tool.correct()
    print("Text after corrected spellings: ", gr1)
    print("\n")

    gr2 = tool2.correct(request.text)
    print("Text after correct grammer 1: ", gr2)
    print("\n")

    gr3 = tool2.correct(gr2)
    print("Text after correct grammer 2: ", gr3)
    print("\n")

    gr4 = tool2.correct(gr3)
    print("Text after correct grammer 3: ", gr4)
    print("\n")

    # For demonstration purposes, returning a dummy response
    response = {"message": "Text processed successfully", "": "", }
    return response
