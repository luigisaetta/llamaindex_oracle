"""
File name: oracle_vector_db.py
Author: Luigi Saetta
Date created: 2023-12-15
Date last modified: 2023-01-05
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
from typing import List, Any, Dict
from llama_index.vector_stores.types import (
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)

# to generate id from text
import hashlib

from llama_index.schema import TextNode, BaseNode

import oracledb
import logging

# load configs from here
from config_private import DB_USER, DB_PWD, DB_HOST_IP, DB_SERVICE

# But for now we don't need to compute the id.. it is set in the driving
# code when the doc list is created
from config import ID_GEN_METHOD, EMBEDDINGS_BITS

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


#
# supporting functions
#
def oracle_query(embed_query: List[float], top_k: int = 2, verbose=False):
    """
    Executes a query against an Oracle database to find the top_k closest vectors to the given embedding.

    History:
        23/12/2023: modified to return some metadata (book_name, page_num)
    Args:
        embed_query (List[float]): A list of floats representing the query vector embedding.
        top_k (int, optional): The number of closest vectors to retrieve. Defaults to 2.
        verbose (bool, optional): If set to True, additional information about the query and execution time will be printed. Defaults to False.

    Returns:
        VectorStoreQueryResult: Object containing the query results, including nodes, similarities, and ids.
    """
    start_time = time.time()

    # build the DSN from data taken from config.py
    DSN = f"{DB_HOST_IP}/{DB_SERVICE}"

    try:
        with oracledb.connect(user=DB_USER, password=DB_PWD, dsn=DSN) as connection:
            with connection.cursor() as cursor:
                # 'f' single precision 'd' double precision
                array_type = "d" if EMBEDDINGS_BITS == 64 else "f"
                array_query = array.array(array_type, embed_query)

                # changed select adding books (39/12/2023)
                select = f"""select V.id, C.CHUNK, C.PAGE_NUM, 
                            ROUND(VECTOR_DISTANCE(V.VEC, :1, DOT), 3) as d,
                            B.NAME 
                            from VECTORS V, CHUNKS C, BOOKS B
                            where C.ID = V.ID and
                            C.BOOK_ID = B.ID
                            order by d
                            FETCH FIRST {top_k} ROWS ONLY"""

                if verbose:
                    logging.info(f"SQL Query: {select}")

                cursor.execute(select, [array_query])
                rows = cursor.fetchall()

                result_nodes, node_ids, similarities = [], [], []

                # prepare output
                for row in rows:
                    # row[1] is a clob
                    full_clob_data = row[1].read()

                    # 29/12: added book_name to metadata
                    result_nodes.append(
                        TextNode(
                            id_=row[0],
                            text=full_clob_data,
                            metadata={"file_name": row[4], "page_label": row[2]},
                        )
                    )
                    node_ids.append(row[0])
                    similarities.append(row[3])

    except Exception as e:
        logging.error(f"Error occurred in oracle_query: {e}")
        return None

    q_result = VectorStoreQueryResult(
        nodes=result_nodes, similarities=similarities, ids=node_ids
    )

    elapsed_time = time.time() - start_time

    if verbose:
        logging.info(f"Query duration: {round(elapsed_time, 1)} sec.")

    return q_result


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


#
# The class wrapping the Oracle DB Vector Store
#
class OracleVectorStore(VectorStore):
    """
    Interface with Oracle DB Vector Store

    """

    stores_text: bool = True
    verbose: bool = False
    DSN = f"{DB_HOST_IP}/{DB_SERVICE}"

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
            # regarding the id, it is already set in the right way
            # is it generated by llama-index, hash or book_name + page_num
            # so we don't need to do anything here
            self.node_dict[node.id_] = node
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
        """
        Persist VectorStore to Oracle DB

        """

        if self.node_dict:
            # not empty, persist
            logging.info("Persisting to DB...")

            embeddings = []
            pages_id = []
            pages_text = []
            pages_num = []

            for key, node in self.node_dict.items():
                pages_id.append(node.id_)
                pages_text.append(node.text)
                embeddings.append(node.embedding)
                pages_num.append(node.metadata["page_label"])

            with oracledb.connect(
                user=DB_USER, password=DB_PWD, dsn=self.DSN
            ) as connection:
                save_embeddings_in_db(embeddings, pages_id, connection)

                # TODO: where should I get book_id?
                save_chunks_in_db(
                    pages_text,
                    pages_id,
                    pages_num=pages_num,
                    book_id=None,
                    connection=connection,
                )

                connection.commit()

            # after persisting empty the cache
            self.node_dict = {}
