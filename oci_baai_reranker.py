"""
File name: oci_baai_reranker.py
Author: Luigi Saetta
Date created: 2023-12-30
Date last modified: 2024-01-02
Python Version: 3.9

Description:
    This module provides the base class to integrate a reranker
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

import cloudpickle
import requests
import base64
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class OCIBAAIReranker:
    def __init__(self, auth, deployment_id, region="eu-frankfurt-1"):
        """
        auth: to manage OCI auth
        deployment_id: the ocid of the model deployment
        region: the OCI region where the deployment is
        top_n: how many to return
        """
        self.auth = auth
        self.deployment_id = deployment_id

        # build the endpoint
        BASE_URL = f"https://modeldeployment.{region}.oci.customer-oci.com/"
        self.endpoint = f"{BASE_URL}{self.deployment_id}/predict"

        logging.info("Created OCI reranker client...")
        logging.info(f"Region: {region}...")
        logging.info(f"Deployment id: {deployment_id}...")
        logging.info("")

    def _build_body(cls, input_list):
        """
        This method builds the body for the https POST call
        """

        # it has been difficult to figure out how to do this... seems to work
        # maybe there are other ways
        val_ser = base64.b64encode(cloudpickle.dumps(input_list)).decode("utf-8")
        body = {"data": val_ser, "data_type": "numpy.ndarray"}

        return body

    @classmethod
    def class_name(cls) -> str:
        return "OCIBAAIReranker"

    def _compute_score(self, x):
        """
        This method exposes the original interface of the Model deployed
        (see BAAI reranker compute_score)
        x: a list of couple of strings to be compared
        example: [["input1", "input2"]]
        """
        # prepares the body for the Model Deployment invocation
        body = self._build_body(x)

        try:
            # here we invoke the deployment
            response = requests.post(self.endpoint, json=body, auth=self.auth["signer"])

            # check if HTTP status is OK
            if response.status_code == 200:
                # ok go forward
                response = response.json()
            else:
                logging.error(
                    f"Error in OCIBAAIReranker compute_score: {response.json()}"
                )
                return []

        except Exception as e:
            logging.error("Error in OCIBAAIReranker compute_score...")
            logging.error(e)

            return []

        return response

    def rerank(self, query, texts, top_n=2):
        """
        Invoke the Model Deployment with the reranker
        - query
        - texts: List[str] are compared and reranked with query
        """
        # BAAI reranker expects input in this way
        # x is a list of list, like [['what is panda?', 'The giant panda is a bear.'],
        #  ['what is panda?', 'It is an animal living in China']]
        x = [[query, text] for text in texts]

        try:
            # here we invoke the deployment
            response = self._compute_score(x)

            # return the texts in order of decreasing score
            # this block of code has been inspired by the code of the cohere_reranker
            sorted_data = []
            if len(response) > 0:
                data = [
                    {"text": text, "index": index, "relevance_score": score}
                    for index, (text, score) in enumerate(
                        zip(texts, response["prediction"])
                    )
                ]

                # sort in decreasing score
                sorted_data = sorted(
                    data, key=lambda x: x["relevance_score"], reverse=True
                )
                # output only top_n
                sorted_data = sorted_data[:top_n]

        except Exception as e:
            logging.error("Error in OCIBAAIReranker rerank...")
            logging.error(e)

            return []

        return sorted_data
