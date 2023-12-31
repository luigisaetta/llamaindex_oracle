"""
File name: config.py
Author: Luigi Saetta
Date created: 2023-12-15
Date last modified: 2023-12-29
Python Version: 3.9

Description:
    This module provides some configurations


Usage:
    Import this module into other scripts to use its functions. 
    Example:
    ...

License:
    This code is released under the MIT License.

Notes:
    This is a part of a set of demo showing how to use Oracle Vector DB,
    OCI GenAI service, Oracle GenAI Embeddings, to build a RAG solution,
    where all he data (text + embeddings) are stored in Oracle DB 23c 

Warnings:
    This module is in development, may change in future versions.
"""

# the book we're going to split and embed
# INPUT_FILES = ["./ambrosetti.pdf"]
# INPUT_FILES = [
#    "database-concepts.pdf",
#    "oracle-database-23c-new-features-guide.pdf",
#    "CurrentEssentialsofMedicine.pdf",
#    "python4everybody.pdf",
#    "Crisidemocraziafakenews.pdf"
# ]
# INPUT_FILES = ["AI Generativa - casi d'uso per industry.pdf"]
INPUT_FILES = [
    "Fare_grafica_editoriale_Progettare_il_libro_storia,_teorie,_tecniche.pdf"
]

# Cohere embeddings model
# for multilingual (es: italian) use this one
EMBED_MODEL = "cohere.embed-multilingual-v3.0"
# used for token counting
TOKENIZER = "Cohere/command-nightly"
# for english use this one
# EMBED_MODEL = "cohere.embed-english-v3.0"

# choose the Gen Model (Mistral to test Italian)
# GEN_MODEL = "OCI"
GEN_MODEL = "MISTRAL"

# for retrieval
TOP_K = 8
# reranker
TOP_N = 4

# for GenAI models
MAX_TOKENS = 1024

# if we want to add a reranker (Cohere for now)
ADD_RERANKER = True
# RERANKER_MODEL = "COHERE"
RERANKER_MODEL = "OCI_BAAI"
RERANKER_ID = "ocid1.datasciencemodeldeployment.oc1.eu-frankfurt-1.amaaaaaangencdyahhhowbtx4blznvebldr73a5dv4m22xjiaux36x4v6sta"

# bits used to store embeddings
# possible values: 32 or 64
# must be aligned with the create_tables.sql used
EMBEDDINGS_BITS = 64

# ID generation: LLINDEX, HASH, BOOK_PAGE_NUM
# define the method to generate ID
ID_GEN_METHOD = "HASH"

# UI
ADD_REFERENCES = True
