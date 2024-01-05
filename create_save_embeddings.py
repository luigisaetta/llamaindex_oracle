"""
File name: create_save_embeddings.py
Author: Luigi Saetta
Date created: 2023-12-14
Date last modified: 2023-12-22
Python Version: 3.9

Description:
    This module provides the code to create and store embeddings and text
    in Oracle DB
    Create embeddings with OCI GenAI, Cohere V3 and loads in Oracle Vector DB

Usage:
    The programs takes all the config from config.py (and secrets from config_private.py)
    Example:
        python create_save_embeddings.py

License:
    This code is released under the MIT License.

Notes:
    This is a part of a set of demo showing how to use Oracle Vector DB,
    OCI GenAI service, Oracle GenAI Embeddings, to build a RAG solution,
    where all he data (text + embeddings) are stored in Oracle DB 23c 

Warnings:
    This module is in development, may change in future versions.
"""

import logging
import re
from tqdm import tqdm
import array
import numpy as np
import time

# to generate id from text
import hashlib

import oci

from llama_index import SimpleDirectoryReader

import oracledb
import ads

# This is the wrapper for GenAI Embeddings
from ads.llm import GenerativeAIEmbeddings

# this way we don't show & share
from config_private import (
    DB_USER,
    DB_PWD,
    DB_SERVICE,
    DB_HOST_IP,
    COMPARTMENT_OCID,
    ENDPOINT,
)

#
# Configs
#
from config import INPUT_FILES, EMBED_MODEL, EMBEDDINGS_BITS, ID_GEN_METHOD

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

    logging.info(f"Read total {len(pages)} pages...")

    # preprocess text
    for doc in pages:
        doc.text = preprocess_text(doc.text)

    # remove pages with num words < threshold
    pages = remove_short_pages(pages, threshold=10)

    # create a list of text (these are the chuncks to be embedded and saved)
    pages_text = [doc.text for doc in pages]

    # 23/12 register the num of the page
    # must be a string
    pages_num = [doc.metadata["page_label"] for doc in pages]

    # extract list of id
    # this way id have been generated by llama-index
    if ID_GEN_METHOD == "LLINDEX":
        pages_id = [doc.id_ for doc in pages]
    # this way generated hashing the page
    if ID_GEN_METHOD == "HASH":
        logging.info("Hashing to compute id...")
        pages_id = []
        for doc in tqdm(pages):
            encoded_text = doc.text.encode()
            hash_object = hashlib.sha256(encoded_text)
            hash_hex = hash_object.hexdigest()
            pages_id.append(hash_hex)

    return pages_text, pages_id, pages_num


# some simple text preprocessing
# TODO: this function must be customized to fit your pdf
def preprocess_text(text):
    text = text.replace("\t", " ")
    text = text.replace(" -\n", "")
    text = text.replace("-\n", "")
    text = text.replace("\n", " ")

    # remove repeated blanks
    text = re.sub(r"\s+", " ", text)

    return text


# remove pages with num words < threshold
def remove_short_pages(pages, threshold):
    n_removed = 0
    for pag in pages:
        if len(pag.text.split(" ")) < threshold:
            pages.remove(pag)
            n_removed += 1

    logging.info(f"Removed {n_removed} short pages...")

    return pages


# take the list of txts and return a list of embeddings vector
def compute_embeddings(embed_model, pages_text):
    embeddings = []
    for i in tqdm(range(0, len(pages_text), BATCH_SIZE)):
        batch = pages_text[i : i + BATCH_SIZE]

        # here we compute embeddings for a batch
        embeddings_batch = embed_model.embed_documents(batch)
        # add to the final list
        embeddings.extend(embeddings_batch)

    return embeddings


def save_embeddings_in_db(embeddings, pages_id, connection):
    tot_errors = 0

    with connection.cursor() as cursor:
        logging.info("Saving embeddings to DB...")

        for id, vector in zip(tqdm(pages_id), embeddings):
            # 'f' single precision 'd' double precision
            if EMBEDDINGS_BITS == 64:
                input_array = array.array("d", vector)
            else:
                # 32 bits
                input_array = array.array("f", vector)

            try:
                # insert single embedding
                cursor.execute("insert into VECTORS values (:1, :2)", [id, input_array])
            except Exception as e:
                logging.error("Error in save embeddings...")
                logging.error(e)
                tot_errors += 1

    logging.info(f"Tot. errors in save_embeddings: {tot_errors}")


# this function is called once for each book
# and saves in DB all the pages of the book + embeddings
def save_chunks_in_db(pages_text, pages_id, pages_num, book_id, connection):
    tot_errors = 0

    with connection.cursor() as cursor:
        logging.info("Saving texts to DB...")
        cursor.setinputsizes(None, oracledb.DB_TYPE_CLOB)

        for id, text, page_num in zip(tqdm(pages_id), pages_text, pages_num):
            try:
                cursor.execute(
                    "insert into CHUNKS (ID, CHUNK, PAGE_NUM, BOOK_ID) values (:1, :2, :3, :4)",
                    [id, text, page_num, book_id],
                )
            except Exception as e:
                logging.error("Error in save chunks...")
                logging.error(e)
                tot_errors += 1

    logging.info(f"Tot. errors in save_chunks: {tot_errors}")


# with this function every book added to DB is registered with a unique id
def register_book(book_name, connection):
    with connection.cursor() as cursor:
        # get the new key
        cursor.execute("SELECT MAX(ID) FROM BOOKS")

        # Fetch the result
        row = cursor.fetchone()

        if row[0] is not None:
            new_key = row[0] + 1
        else:
            new_key = 1

    # insert the record for the book
    with connection.cursor() as cursor:
        query = "INSERT INTO BOOKS (ID, NAME) VALUES (:1, :2)"

        # Execute the query with your values
        cursor.execute(query, [new_key, book_name])

    return new_key


#
# Main
#

# mark start
tStart = time.time()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

print("")
print("Start processing...")
print("")
print("List of books to be loaded and indexed:")

# print list of book to be loaded
for book_name in INPUT_FILES:
    print(book_name)
print("")

oci_config = load_oci_config()

# need to do this way
api_keys_config = ads.auth.api_keys(oci_config)

# the embedding client
embed_model = GenerativeAIEmbeddings(
    compartment_id=COMPARTMENT_OCID,
    model=EMBED_MODEL,
    auth=api_keys_config,
    # Optionally you can specify keyword arguments for the OCI client, e.g. service_endpoint.
    client_kwargs={"service_endpoint": ENDPOINT},
)

# connect to db
logging.info("Connecting to Oracle DB...")

DSN = DB_HOST_IP + "/" + DB_SERVICE

with oracledb.connect(user=DB_USER, password=DB_PWD, dsn=DSN) as connection:
    logging.info("Successfully connected to Oracle Database...")

    num_pages = []
    for book in INPUT_FILES:
        logging.info(f"Processing book: {book}...")
        # chunks are pages
        pages_text, pages_id, pages_num = read_and_split_in_pages([book])
        num_pages.append(len(pages_text))

        # create embeddings
        # process in batch (max 96 for batch, chosen BATCH_SIZE, see above)
        logging.info("Computing embeddings...")

        embeddings = compute_embeddings(embed_model, pages_text)

        # determine book_id and save in table BOOKS
        logging.info("Registering book...")

        book_id = register_book(book, connection)

        # store embeddings
        # here we save in DB
        save_embeddings_in_db(embeddings, pages_id, connection)

        logging.info("Save embeddings OK...")

        # store text chunks (pages for now)
        save_chunks_in_db(pages_text, pages_id, pages_num, book_id, connection)

        # a txn is a book
        connection.commit()

        logging.info("Save texts OK...")

    # end !!!
    tot_pages = np.sum(np.array(num_pages))

tEla = time.time() - tStart

print("")
print("Processing done !!!")
print(
    f"We have processed {tot_pages} pages and saved text chunks and embeddings in the DB"
)
print(f"Total elapsed time: {round(tEla, 0)} sec.")
print()
