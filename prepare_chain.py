"""
File name: prepare_chain.py
Author: Luigi Saetta
Date created: 2023-12-17
Date last modified: 2023-01-05
Python Version: 3.9

Description:
    This module provides a function to initialize the RAG chain 

Usage:
    Import this module into other scripts to use its functions. 
    Example:
        from prepare_chain import create_query_engine

License:
    This code is released under the MIT License.

Notes:
    This is a part of a set of demo showing how to use Oracle Vector DB,
    OCI GenAI service, Oracle GenAI Embeddings, to buil a RAG solution,
    where all he data (text + embeddings) are stored in Oracle DB 23c 

    Now it can use for LLM: OCI, Mistral 8x7B

Warnings:
    This module is in development, may change in future versions.
"""

import logging

from llama_index import VectorStoreIndex, ServiceContext
from llama_index.callbacks import CallbackManager
from tokenizers import Tokenizer
from llama_index.callbacks import TokenCountingHandler
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.llms import MistralAI

import ads
from ads.llm import GenerativeAIEmbeddings, GenerativeAI

# COHERE_KEY is used for reranker
# MISTRAL_KEY for LLM
from config_private import COMPARTMENT_OCID, ENDPOINT, MISTRAL_API_KEY, COHERE_API_KEY

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
)

from oci_utils import load_oci_config, print_configuration
from oracle_vector_db import OracleVectorStore
from oci_baai_reranker import OCIBAAIReranker
from oci_llama_reranker import OCILLamaReranker

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

#
# This module now expse directly the factory methods for all the single components (llm, etc)
# the philosophy of the factory methods  is that they're taking all the infos from the config
# module... so as few parameter as possible
#


#
# enables to plug different GEN_MODELS
# for now: OCI, MISTRAL
#
def create_llm(auth=None):
    llm = None

    if GEN_MODEL == "OCI":
        llm = GenerativeAI(
            auth=auth,
            compartment_id=COMPARTMENT_OCID,
            max_tokens=MAX_TOKENS,
            # added 23/12 to avoid error for context too long
            truncate="END",
            client_kwargs={"service_endpoint": ENDPOINT},
        )
    if GEN_MODEL == "MISTRAL":
        llm = MistralAI(
            api_key=MISTRAL_API_KEY,
            model="mistral-small",
            temperature=0.2,
            max_tokens=MAX_TOKENS,
        )

    return llm


def create_reranker(auth=None, verbose=False):
    reranker = None

    if RERANKER_MODEL == "COHERE":
        reranker = CohereRerank(api_key=COHERE_API_KEY, top_n=TOP_N)

    if RERANKER_MODEL == "OCI_BAAI":
        baai_reranker = OCIBAAIReranker(
            auth=auth, deployment_id=RERANKER_ID, region="eu-frankfurt-1"
        )

        reranker = OCILLamaReranker(
            oci_reranker=baai_reranker, top_n=TOP_N, verbose=verbose
        )

    return reranker


def create_embedding_model(auth=None):
    embed_model = None

    if EMBED_MODEL_TYPE == "OCI":
        embed_model = GenerativeAIEmbeddings(
            compartment_id=COMPARTMENT_OCID,
            model=EMBED_MODEL,
            auth=auth,
            # Optionally you can specify keyword arguments for the OCI client
            # e.g. service_endpoint.
            client_kwargs={"service_endpoint": ENDPOINT},
        )

    return embed_model


def create_query_engine(token_counter=None, verbose=False):
    logging.info("calling create_query_engine()...")

    print_configuration()

    # load security info needed for OCI
    oci_config = load_oci_config()

    # need to do this way
    api_keys_config = ads.auth.api_keys(oci_config)

    # this is to embed the question
    embed_model = create_embedding_model(auth=api_keys_config)

    # this is the custom class to access Oracle DB as Vectore Store
    v_store = OracleVectorStore(verbose=False)

    # this is to access OCI or MISTRAL GenAI service
    llm = create_llm(auth=api_keys_config)

    # this part has been added to count the total # of tokens
    cohere_tokenizer = Tokenizer.from_pretrained(TOKENIZER)
    token_counter = TokenCountingHandler(tokenizer=cohere_tokenizer.encode)

    callback_manager = CallbackManager([token_counter])

    # integrate OCI/Mistral in llama-index
    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model=embed_model, callback_manager=callback_manager
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=v_store, service_context=service_context
    )

    # the whole chain (query string -> embed query -> retrieval -> context, query-> GenAI -> response)
    # is wrapped in the query engine

    # here we could plug a reranker improving the quality
    if ADD_RERANKER == True:
        reranker = create_reranker(auth=api_keys_config)

        query_engine = index.as_query_engine(
            similarity_top_k=TOP_K, node_postprocessors=[reranker]
        )
    else:
        query_engine = index.as_query_engine(similarity_top_k=TOP_K)

    # to add a blank line in the log
    logging.info("")

    return query_engine, token_counter
