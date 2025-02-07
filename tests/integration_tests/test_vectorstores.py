import logging
import os
import uuid
from typing import Generator

import pytest
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import VectorStoreIntegrationTests

from langchain_vdms.vectorstores import VDMS, VDMS_Client

logging.basicConfig(level=logging.DEBUG)

# To spin up a detached VDMS server:
# docker pull intellabs/vdms:latest
# docker run -d -p $VDMS_DBPORT:55555 intellabs/vdms:latest


class TestVdmsVectorStoreSync(VectorStoreIntegrationTests):
    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore
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
