#!/usr/bin/env python
# coding: utf-8
# ######################################################### Document search using  Milvus  #################################################################
# 
# In this example, we will download text and embeddings using the LANGCHAIN API we can then query that body of work
# the langchain package well embed the query the same way by default 
# 
# https://www.langchain.com/
# 
# The process is as follows
# 1) Load the document you want to search thru
# 2) split that data into chunks
#    ---------------retrieve embedding model
# 3) Vectorize the text chunks using this model
# 4) feed a query and this doc into chain()
######################################################################################################################################################################

# 
#!pip3 install langchain openai tiktoken

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Milvus
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from os import environ
from getpass import getpass




# load the data
loader = TextLoader("chess_wc_2023.txt")
docs = loader.load()




# Split the data into multiple non-overlapping chunks

text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
docs = text_splitter.split_documents(docs)



## Set up OPENAI_API_KEY environment variables

OPENAI_API_KEY = getpass('OpenAPI API Key: ')  
environ["OPENAI_API_KEY"] = OPENAI_API_KEY



# Declare the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")


# Declare the vector store, and store the embeddings from the doc

vector_store = Milvus.from_documents(
    docs,
    embedding=embeddings,
    connection_args={"host": "localhost", "port": "19530"}
)


chain = load_qa_with_sources_chain(OpenAI(temperature=0), chain_type="map_reduce", return_intermediate_steps=False)
query = "What is the prize money?"
chain({"input_documents": docs, "question": query}, return_only_outputs=True)



