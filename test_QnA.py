import openai
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity
import copy
import pickle

def get_embedding_from_text(text):
    embedding = get_embedding(
        text,
        engine="text-embedding-ada-002"
    )
    return embedding

def text_search(text, df):
    embedding = get_embedding_from_text(text)

    for item in df:
      df["similarities"] = df.Embeddings.apply(lambda x: cosine_similarity(x, embedding))
    return df

def get_top_reply(text, df, n=1, minimum_similarity=0.85):
    df = copy.copy(df)
    df = text_search(text, df)
    df = df.sort_values("similarities", ascending=False).head(n)
    if df.iloc[0]["similarities"] < minimum_similarity:
        top_question = "?"
        top_answer = "Sorry, I don't know the answer to that question."
    else:
        top_question = df["Questions"].iloc[0]
        top_answer = df["Answers"].iloc[0]

    return top_answer


def test_QnA(path, openai_key, num_replies=4, minimum_similarity=0.85):
    
    openai.api_key = openai_key
    
    df = pickle.load(open(path, "rb"))
    
    for i in range(num_replies):
      reply = get_top_reply(input("Message: "), df,minimum_similarity=minimum_similarity)
      print("Reply:", reply)
