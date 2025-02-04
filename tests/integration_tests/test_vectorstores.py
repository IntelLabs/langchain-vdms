from typing import AsyncGenerator, Generator

import pytest
from langchain_vdms.vectorstores import VdmsVectorStore
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import (
    AsyncReadWriteTestSuite,
    ReadWriteTestSuite,
)


class TestVdmsVectorStoreSync(ReadWriteTestSuite):
    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore
        """Get an empty vectorstore for unit tests."""
        store = VdmsVectorStore()
        # note: store should be EMPTY at this point
        # if you need to delete data, you may do so here
        try:
            yield store
        finally:
            # cleanup operations, or deleting data
            pass


class TestVdmsVectorStoreAsync(AsyncReadWriteTestSuite):
    @pytest.fixture()
    async def vectorstore(self) -> AsyncGenerator[VectorStore, None]:  # type: ignore
        """Get an empty vectorstore for unit tests."""
        store = VdmsVectorStore()
        # note: store should be EMPTY at this point
        # if you need to delete data, you may do so here
        try:
            yield store
        finally:
            # cleanup operations, or deleting data
            pass
