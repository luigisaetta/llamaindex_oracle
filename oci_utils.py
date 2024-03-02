"""
File name: oci_utils.py
Author: Luigi Saetta
Date created: 2023-12-17
Date last modified: 2023-12-17
Python Version: 3.9

Description:
    This module provides some utilities

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

import logging
import oci
from config import (
    EMBED_MODEL_TYPE,
    EMBED_MODEL,
    TOKENIZER,
    GEN_MODEL,
    MAX_TOKENS,
    TOP_K,
    ADD_RERANKER,
    RERANKER_MODEL,
    RERANKER_ID,
    TOP_N,
    ADD_PHX_TRACING,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_oci_config():
    # read OCI config to connect to OCI with API key

    # are you using default profile?
    oci_config = oci.config.from_file("~/.oci/config", "DEFAULT")

    return oci_config


def print_configuration():
    logging.info("------------------------")
    logging.info("Config. used:")
    logging.info(f"{EMBED_MODEL_TYPE} {EMBED_MODEL} for embeddings...")
    logging.info("Using Oracle DB Vector Store...")
    logging.info(f"Using {GEN_MODEL} as LLM...")
    logging.info("Retrieval parameters:")
    logging.info(f"TOP_K: {TOP_K}")

    if ADD_RERANKER:
        logging.info(f"Using {RERANKER_MODEL} as reranker...")
        logging.info(f"TOP_N: {TOP_N}")
    if ADD_PHX_TRACING:
        logging.info(f"Enabled observability with Phoenix tracing...")

    logging.info("------------------------")
    logging.info("")


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
