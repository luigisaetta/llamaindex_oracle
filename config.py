"""
File name: config.py
Author: Luigi Saetta
Date created: 2023-12-15
Date last modified: 2023-12-22
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
INPUT_FILES = ["ai-4-italy.pdf",
               "feynman_vol1.pdf"]

# Cohere embeddings model
# for english use this one
EMBED_MODEL = "cohere.embed-multilingual-v3.0"
# used for token counting
TOKENIZER = "Cohere/command-nightly"
# for other language (must be consistent when do query)
# EMBED_MODEL = "cohere.embed-multilingual-v3.0"

# choose the Gen Model (Mistral to test Italian)
# GEN_MODEL = "OCI"
GEN_MODEL = "MISTRAL"

# bits used to store embeddings
# possible values: 32 or 64
# must be aligned with the create_tables.sql used
EMBEDDINGS_BITS = 64

# ID generation: LLINDEX, HASH, BOOK_PAGE_NUM
# define the method to generate ID
ID_GEN_METHOD = "HASH"
