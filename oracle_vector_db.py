"""
File name: oracle_vector_db.py
Author: Luigi Saetta
Date created: 2023-12-15
Date last modified: 2023-12-17
Python Version: 3.9

Description:
    This module provides the class to integrate Oracle Vector DB 
    as Vector Store in llama-index

Inspired by:
    https://docs.llamaindex.ai/en/stable/examples/low_level/vector_store.html

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

import time
from tqdm import tqdm
import array
from typing import List, Any, Tuple, Dict
from llama_index.vector_stores.types import (
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.schema import TextNode, BaseNode

import oracledb
import logging

# load configs from here
from config_private import DB_USER, DB_PWD, DB_HOST_IP, DB_SERVICE

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def oracle_query(embed_query: List[float], top_k: int = 2, verbose=False):
    """
    Executes a query against an Oracle database to find the top_k closest vectors to the given embedding.

    Args:
        embed_query (List[float]): A list of floats representing the query vector embedding.
        top_k (int, optional): The number of closest vectors to retrieve. Defaults to 2.
        verbose (bool, optional): If set to True, additional information about the query and execution time will be printed. Defaults to False.

    Returns:
        VectorStoreQueryResult: Object containing the query results, including nodes, similarities, and ids.
    """
    tStart = time.time()

    DSN = DB_HOST_IP + "/" + DB_SERVICE

    try:
        with oracledb.connect(user=DB_USER, password=DB_PWD, dsn=DSN) as connection:
            with connection.cursor() as cursor:
                array_query = array.array("d", embed_query)

                select = f"""select V.id, C.CHUNK, ROUND(VECTOR_DISTANCE(V.VEC, :1, DOT), 3)
                            as d from VECTORS V, CHUNKS C
                            where C.ID = V.ID
                            order by d
                            FETCH FIRST {top_k} ROWS ONLY"""

                if verbose:
                    print(f"select: {select}")

                cursor.execute(select, [array_query])

                rows = cursor.fetchall()

                result_nodes = []
                node_ids = []
                similarities = []

                # prepare output
                for row in rows:
                    clob_pointer = row[1]
                    full_clob_data = clob_pointer.read()

                    result_nodes.append(TextNode(id_=row[0], text=full_clob_data))
                    node_ids.append(row[0])
                    similarities.append(row[2])

    except Exception as e:
        logging.error(f"Error occurred in oracle_query: {e}")

        return None

    q_result = VectorStoreQueryResult(
        nodes=result_nodes, similarities=similarities, ids=node_ids
    )

    tEla = time.time() - tStart

    if verbose:
        logging.info(f"Query duration: {round(tEla, 1)} sec.")

    return q_result


def save_embeddings_in_db(embeddings, pages_id, connection):
    with connection.cursor() as cursor:
        logging.info("Saving embeddings to DB...")

        for id, vector in zip(tqdm(pages_id), embeddings):
            # to handle 64 bit correctly
            input_array = array.array("d", vector)

            cursor.execute("insert into VECTORS values (:1, :2)", [id, input_array])


def save_chunks_in_db(pages_text, pages_id, connection):
    with connection.cursor() as cursor:
        logging.info("Saving texts to DB...")
        cursor.setinputsizes(None, oracledb.DB_TYPE_CLOB)

        for id, text in zip(tqdm(pages_id), pages_text):
            cursor.execute("insert into CHUNKS values (:1, :2)", [id, text])


class OracleVectorStore(VectorStore):
    """
    Interface with Oracle DB Vector Store

    """

    stores_text: bool = True
    verbose: bool = False
    DSN = DB_HOST_IP + "/" + DB_SERVICE

    def __init__(self, verbose=False) -> None:
        """Init params."""
        self.verbose = verbose

        # initialize the cache
        self.node_dict: Dict[str, BaseNode] = {}

    # get method is NOT needed

    def add(
        self,
        nodes: List[BaseNode],
    ) -> List[str]:
        """Add nodes to index."""
        ids_list = []
        for node in nodes:
            # the node contains already the embedding
            self.node_dict[node.node_id] = node
            ids_list.append(node.id_)

        return ids_list

    def delete(self, node_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        raise NotImplementedError("Delete not yet implemented for Oracle Vector Store.")

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Get nodes for response."""

        if self.verbose:
            logging.info("---> Calling query on DB")

        return oracle_query(
            query.query_embedding, top_k=query.similarity_top_k, verbose=self.verbose
        )

    def persist(self, persist_path=None, fs=None) -> None:
        """Persist the SimpleVectorStore to a directory.

        NOTE: we are not implementing this for now.

        """

        if self.node_dict:
            # not empty, persist
            logging.info("Persisting to DB...")

            embeddings = []
            pages_id = []
            pages_text = []

            for key, node in self.node_dict.items():
                pages_id.append(node.id_)
                pages_text.append(node.text)
                embeddings.append(node.embedding)

            with oracledb.connect(
                user=DB_USER, password=DB_PWD, dsn=self.DSN
            ) as connection:
                save_embeddings_in_db(embeddings, pages_id, connection)

                save_chunks_in_db(pages_text, pages_id, connection)

                connection.commit()

            # after persisting empty the cache
            self.node_dict = {}
