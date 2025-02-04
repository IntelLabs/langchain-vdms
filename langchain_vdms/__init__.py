from importlib import metadata

from langchain_vdms.chat_models import ChatVdms
from langchain_vdms.document_loaders import VdmsLoader
from langchain_vdms.embeddings import VdmsEmbeddings
from langchain_vdms.retrievers import VdmsRetriever
from langchain_vdms.toolkits import VdmsToolkit
from langchain_vdms.tools import VdmsTool
from langchain_vdms.vectorstores import VdmsVectorStore

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatVdms",
    "VdmsVectorStore",
    "VdmsEmbeddings",
    "VdmsLoader",
    "VdmsRetriever",
    "VdmsToolkit",
    "VdmsTool",
    "__version__",
]
