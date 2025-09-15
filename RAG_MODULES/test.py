from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
load_dotenv()
api=os.getenv("PINECONE")
pc = Pinecone(api_key=api)

# from pinecone import Pinecone

# pc = Pinecone(api_key="********-****-****-****-************")
index = pc.Index("saarathi-finance")


