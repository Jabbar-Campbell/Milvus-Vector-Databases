#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip3 install langchain openai tiktoken

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Milvus
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from os import environ
from getpass import getpass


# In[2]:


# load the data
loader = TextLoader("chess_wc_2023.txt")
docs = loader.load()


# In[3]:


# Split the data into multiple non-overlapping chunks

text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
docs = text_splitter.split_documents(docs)


# In[4]:


## Set up OPENAI_API_KEY environment variables

OPENAI_API_KEY = getpass('OpenAPI API Key: ')  
environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# In[5]:


# Declare the embedding model

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")


# In[6]:


# Declare the vector store, and store the embeddings from the doc

vector_store = Milvus.from_documents(
    docs,
    embedding=embeddings,
    connection_args={"host": "localhost", "port": "19530"}
)


# In[11]:


chain = load_qa_with_sources_chain(OpenAI(temperature=0), chain_type="map_reduce", return_intermediate_steps=False)
query = "What is the prize money?"
chain({"input_documents": docs, "question": query}, return_only_outputs=True)


# In[ ]:




