import logging
import os
import uuid

import pytest
from langchain_core.documents import Document
from langchain_tests.integration_tests import VectorStoreIntegrationTests

from langchain_vdms.vectorstores import VDMS, VDMS_Client

logging.basicConfig(level=logging.DEBUG)

# To spin up a detached VDMS server:
# docker pull intellabs/vdms:latest
# docker run -d -p $VDMS_DBPORT:55555 intellabs/vdms:latest


class TestVdmsVectorStoreSync(VectorStoreIntegrationTests):
    @pytest.fixture()
    def vectorstore(self) -> VDMS:  # type: ignore
        """Get an empty vectorstore for unit tests."""
        test_name = uuid.uuid4().hex
        client = VDMS_Client(
            host=os.getenv("VDMS_DBHOST", "localhost"),
            port=int(os.getenv("VDMS_DBPORT", 6025)),
        )
        store = VDMS(
            client=client,
            embedding=self.get_embeddings(),
            collection_name=test_name,
        )
        # note: store should be EMPTY at this point
        # if you need to delete data, you may do so here
        try:
            yield store
        finally:
            # cleanup operations, or deleting data
            # logic is executed in between each test
            pass

    @property
    def has_async(self) -> bool:
        """
        Configurable property to enable or disable async tests.
        """
        return False

    @pytest.mark.xfail(
        reason="add_documents can duplicate ids; upsert used for idempotent ids"
    )
    def test_add_documents_with_ids_is_idempotent(self, vectorstore: VDMS) -> None:  # type: ignore[override]
        """Adding by ID should be idempotent.

        .. dropdown:: Troubleshooting

            If this test fails, check that adding the same document twice with the
            same IDs has the same effect as adding it once (i.e., it does not
            duplicate the documents).
        """
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        vectorstore.add_documents(documents, ids=["1", "2"])
        vectorstore.upsert(documents, ids=["1", "2"])
        documents = vectorstore.similarity_search("bar", k=2)
        assert documents == [
            Document(page_content="bar", metadata={"id": 2}, id="2"),
            Document(page_content="foo", metadata={"id": 1}, id="1"),
        ]
