# langchain-vdms

This package contains the LangChain integration with Vdms

## Installation

```bash
pip install -U langchain-vdms
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatVdms` class exposes chat models from Vdms.

```python
from langchain_vdms import ChatVdms

llm = ChatVdms()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`VdmsEmbeddings` class exposes embeddings from Vdms.

```python
from langchain_vdms import VdmsEmbeddings

embeddings = VdmsEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`VdmsLLM` class exposes LLMs from Vdms.

```python
from langchain_vdms import VdmsLLM

llm = VdmsLLM()
llm.invoke("The meaning of life is")
```
