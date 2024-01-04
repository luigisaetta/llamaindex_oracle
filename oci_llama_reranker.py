"""
File name: oci_llama_reranker.py
Author: Luigi Saetta
Date created: 2023-12-30
Date last modified: 2024-01-02
Python Version: 3.9

Description:
    This module provides the class to integrate a reranker
    deployed as Model Deployment in OCI Data Science 
    as reranker in llama-index

Inspired by:
    https://github.com/run-llama/llama_index/blob/main/llama_index/postprocessor/cohere_rerank.py

Usage:
    Import this module into other scripts to use its functions. 
    Example:
    baai_reranker = OCIBAAIReranker(
            auth=api_keys_config, 
            deployment_id=RERANKER_ID, region="eu-frankfurt-1")
        
    reranker = OCILLamaReranker(oci_reranker=baai_reranker, top_n=TOP_N)

License:
    This code is released under the MIT License.

Notes:
    This is a part of a set of demo showing how to use Oracle Vector DB,
    OCI GenAI service, Oracle GenAI Embeddings, to build a RAG solution,
    where all he data (text + embeddings) are stored in Oracle DB 23c 

Warnings:
    This module is in development, may change in future versions.
"""

import time
from typing import Any, List, Optional

from llama_index.bridge.pydantic import Field
from llama_index.callbacks import CBEventType, EventPayload
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.schema import NodeWithScore, QueryBundle
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class OCILLamaReranker(BaseNodePostprocessor):
    # by default use BAAI reranker large, deployed as OCI ODS model
    model: str = "oci_baai_reranker"
    top_n: int = 2
    oci_reranker: Any = None
    verbose: bool = False

    def __init__(
        self,
        model: str = "oci_baai_reranker",
        top_n: int = 2,
        oci_reranker: Any = None,
        verbose: bool = False,
    ) -> None:
        # this one to store model and top_n
        super().__init__(top_n=top_n, model=model)

        self.oci_reranker = oci_reranker
        self.verbose = verbose

    @classmethod
    def class_name(cls) -> str:
        return "OCILLamaReranker"

    def _rerank(self, query, texts, top_n=2):
        return self.oci_reranker.rerank(query, texts, top_n)

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        tStart = time.time()
        logging.info("Reranking...")

        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.MODEL_NAME: self.model,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:
            # extract texts from node list
            texts = [node.node.get_content() for node in nodes]

            results = self._rerank(
                query=query_bundle.query_str, texts=texts, top_n=self.top_n
            )

            # build the output list to be compatible with llama-index
            new_nodes = []
            for result in results:
                new_node_with_score = NodeWithScore(
                    node=nodes[result["index"]].node, score=result["relevance_score"]
                )
                new_nodes.append(new_node_with_score)
            event.on_end(payload={EventPayload.NODES: new_nodes})

        tEla = time.time() - tStart
        if self.verbose:
            logging.info(f"...elapsed time: {round(tEla, 2)} sec.")
        return new_nodes
