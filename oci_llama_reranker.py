import os
from typing import Any, List, Optional

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks import CBEventType, EventPayload
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.schema import NodeWithScore, QueryBundle


class OCILLamaReranker(BaseNodePostprocessor):
    # by default use BAAY reranker large, deployed as OCI ODS model
    model: str = "oci_baai_reranker"
    top_n: int = 2
    oci_reranker: Any = None

    def __init__(
        self, model: str = "oci_baai_reranker", top_n: int = 2, oci_reranker: Any = None
    ) -> None:
        # this one to store model and top_n
        super().__init__(top_n=top_n, model=model)

        self.oci_reranker = oci_reranker

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
            texts = [node.node.get_content() for node in nodes]

            results = self._rerank(
                query=query_bundle.query_str, texts=texts, top_n=self.top_n
            )

            new_nodes = []
            for result in results:
                new_node_with_score = NodeWithScore(
                    node=nodes[result["index"]].node, score=result["relevance_score"]
                )
                new_nodes.append(new_node_with_score)
            event.on_end(payload={EventPayload.NODES: new_nodes})

        return new_nodes
