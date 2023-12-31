import cloudpickle
import requests
import numpy as np
import base64
import ads
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

    def _build_body(cls, input_list):
        """
        This method builds the body for the https POST call
        """
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
            response = requests.post(
                self.endpoint, json=body, auth=self.auth["signer"]
            ).json()

        except Exception as e:
            logging.error("Error in OCIReranker compute_score...")
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

        # prepares the body for the Model Deployment invocation
        body = self._build_body(x)

        try:
            # here we invoke the deployment
            response = self._compute_score(x)

            # return the texts in order of decreasing score
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
            logging.error("Error in OCIReranker rerank...")
            logging.error(e)

            return []

        return sorted_data
