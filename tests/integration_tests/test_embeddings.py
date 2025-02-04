"""Test Vdms embeddings."""

from typing import Type

from langchain_vdms.embeddings import VdmsEmbeddings
from langchain_tests.integration_tests import EmbeddingsIntegrationTests


class TestParrotLinkEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[VdmsEmbeddings]:
        return VdmsEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "nest-embed-001"}
