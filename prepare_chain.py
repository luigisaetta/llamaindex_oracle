import logging
import sys

from typing import List, Any, Optional, Dict, Tuple
from llama_index.vector_stores.types import (
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index import StorageContext, VectorStoreIndex, ServiceContext
from llama_index.schema import TextNode, BaseNode, Document

import oci
import ads
from ads.llm import GenerativeAIEmbeddings, GenerativeAI

from config_private import COMPARTMENT_OCID, ENDPOINT

from oci_utils import load_oci_config
from oracle_vector_db import OracleVectorStore


def create_query_engine():
    oci_config = load_oci_config()

    # need to do this way
    api_keys_config = ads.auth.api_keys(oci_config)

    # english, or for other language use: multilingual
    MODEL_NAME = "cohere.embed-english-v3.0"

    embed_model = GenerativeAIEmbeddings(
        compartment_id=COMPARTMENT_OCID,
        model=MODEL_NAME,
        auth=ads.auth.api_keys(oci_config),
        # Optionally you can specify keyword arguments for the OCI client, e.g. service_endpoint.
        client_kwargs={"service_endpoint": ENDPOINT},
    )

    v_store = OracleVectorStore(verbose=False)

    llm_oci = GenerativeAI(
        compartment_id=COMPARTMENT_OCID,
        max_tokens=1024,
        # Optionally you can specify keyword arguments for the OCI client, e.g. service_endpoint.
        client_kwargs={"service_endpoint": ENDPOINT},
    )

    service_context = ServiceContext.from_defaults(llm=llm_oci, embed_model=embed_model)

    index = VectorStoreIndex.from_vector_store(
        vector_store=v_store, service_context=service_context
    )

    query_engine = index.as_query_engine(similarity_top_k=5)

    return query_engine
