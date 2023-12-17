"""
File name: oracle_vector_db.py
Author: Luigi Saetta
Date created: 2023-12-15
Date last modified: 2023-12-17
Python Version: 3.9

Description:
    This module provides the class to integrate Oracle Vector DB 
    as Vector Store in llama-index 

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
import array
from typing import List, Any, Tuple
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

    def get(self, text_id: str) -> List[float]:
        """Get embedding."""
        try:
            with oracledb.connect(
                user=DB_USER, password=DB_PWD, dsn=self.DSN
            ) as connection:
                with connection.cursor() as cursor:
                    select = f"""select V.id, V.vec
                            from VECTORS V
                            where V.id = :1"""

                    cursor.execute(select, [text_id])

                    # expected 0 or 1 row
                    row = cursor.fetchone()

                    if row:
                        embed_vec = row[1]
                    else:
                        # empty list, is OK?
                        embed_vec = []

            return list(embed_vec)

        except Exception as e:
            logging.error(f"Error occurred in get: {e}")

            return None

    def add(
        self,
        nodes: List[BaseNode],
    ) -> List[str]:
        """Add nodes to index."""
        raise NotImplementedError("This feature is not yet implemented")

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        raise NotImplementedError("This feature is not yet implemented")

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

    def persist(self, persist_path, fs=None) -> None:
        """Persist the SimpleVectorStore to a directory.

        NOTE: we are not implementing this for now.

        """
        raise NotImplementedError("This feature is not yet implemented")
