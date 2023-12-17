"""
File name: create_save_embeddings.py
Author: Luigi Saetta
Date created: 2023-12-14
Date last modified: 2023-12-17
Python Version: 3.9

Description:
    This module provides the code to create and store embeddings and text
    in Oracle DB
    Create embeddings with OCI GenAI, Cohere V3 and loads in Oracle Vector DB

Usage:
    Import this module into other scripts to use its functions. 
    Example:
    ...

License:
    This code is released under the MIT License.

Notes:
    This is a part of a set of demo showing how to use Oracle Vector DB,
    OCI GenAI service, Oracle GenAI Embeddings, to buil a RAG solution,
    where all he data (text + embeddings) are stored in Oracle DB 23c 

Warnings:
    This module is in development, may change in future versions.
"""

import re
import numpy as np
from tqdm import tqdm
import array

import oci

from llama_index import SimpleDirectoryReader

import oracledb
import ads

# This is the wrapper for GenAI Embeddings
from ads.llm import GenerativeAIEmbeddings

# this way we don't show & share
from config_private import DB_USER, DB_PWD, DB_SERVICE, DB_HOST_IP, COMPARTMENT_OCID

#
# Configs
#
# INPUT_FILES = ["./ambrosetti.pdf"]
INPUT_FILES = ["./database-concepts.pdf", "oracle-database-23c-new-features-guide.pdf"]


# OCI settings
ENDPOINT = "https://generativeai.aiservice.us-chicago-1.oci.oraclecloud.com"

# english, for other language use: multilingual
# EMBED_MODEL = "cohere.embed-multilingual-v3.0"
EMBED_MODEL = "cohere.embed-english-v3.0"

# to create embeddings in batch
BATCH_SIZE = 20

#
# Functions
#


# to handle security
def load_oci_config():
    # read OCI config to connect to OCI with API key

    # are you using default profile?
    oci_config = oci.config.from_file("~/.oci/config", "DEFAULT")

    return oci_config


def read_and_split_in_pages(input_files):
    pages = SimpleDirectoryReader(input_files=input_files).load_data()

    print(f"Read total {len(pages)} pages...")

    # preprocess text
    for doc in pages:
        doc.text = preprocess_text(doc.text)

    # remove pages with num words < threshold
    pages = remove_short_pages(pages, threshold=10)

    # create a list of text (these are the chuncks to be embedded and saved)
    pages_text = [doc.text for doc in pages]

    # extract list of id
    pages_id = [doc.id_ for doc in pages]

    return pages_text, pages_id


# some simple text preprocessing
def preprocess_text(text):
    text = text.replace("\t", " ")
    text = text.replace(" -\n", "")
    text = text.replace("-\n", "")
    text = text.replace("\n", " ")

    # remove copyright
    text = text.replace("© The European House - Ambrosetti", " ")
    # remove repeated blanks
    text = re.sub(r"\s+", " ", text)

    return text


# remove pages with num words < threshold
def remove_short_pages(pages, threshold):
    for pag in pages:
        if len(pag.text.split(" ")) < threshold:
            pages.remove(pag)

    return pages


#
# Main
#

print("")
print("Start processing...")
print("")

oci_config = load_oci_config()

# need to do this way
api_keys_config = ads.auth.api_keys(oci_config)

# load books
# chunks are pages
pages_text, pages_id = read_and_split_in_pages(INPUT_FILES)

# create embeddings
embed_model = GenerativeAIEmbeddings(
    compartment_id=COMPARTMENT_OCID,
    model=EMBED_MODEL,
    auth=ads.auth.api_keys(oci_config),
    # Optionally you can specify keyword arguments for the OCI client, e.g. service_endpoint.
    client_kwargs={"service_endpoint": ENDPOINT},
)

embeddings = []

# to process in batch (max 96 for batch)
print("Computing embeddings...")

for i in tqdm(range(0, len(pages_text), BATCH_SIZE)):
    batch = pages_text[i : i + BATCH_SIZE]

    embeddings_batch = embed_model.embed_documents(batch)
    # add to the final list
    embeddings.extend(embeddings_batch)

# save in DB

# connect to db
print("Connecting to Oracle DB...")

connection = oracledb.connect(
    user=DB_USER, password=DB_PWD, dsn=DB_HOST_IP + "/" + DB_SERVICE
)

print("Successfully connected to Oracle Database...")

# store embeddings
cursor = connection.cursor()

print("Saving embeddings to DB...")
i = 0
for id, vector in zip(tqdm(pages_id), embeddings):
    i += 1
    # to handle 64 bit corrctly
    input_array = array.array("d", vector)

    cursor.execute("insert into VECTORS values (:1, :2)", [id, input_array])
    # moved in the loop to save resource in the db... can be slower
    connection.commit()
cursor.close()

print("Save OK...")

# store text chunks (pages for now)
cursor = connection.cursor()

cursor.setinputsizes(None, oracledb.DB_TYPE_CLOB)

print("Saving chunks to DB...")
i = 0
for id, text in zip(tqdm(pages_id), pages_text):
    i += 1
    cursor.execute("insert into CHUNKS values (:1, :2)", [id, text])
    connection.commit()
cursor.close()

print("Save OK...")

# end !!!
connection.close()

print("")
print("Processing done !!!")
print(
    f"We have processed {len(pages_text)} pages and saved chunks and embeddings in the DB"
)
print()
