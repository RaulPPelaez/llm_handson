# Some usecase examples with local LLMs

Open the different notebooks and follow along. The intended order is:

1. `huggingface.ipynb`: How to use a LLM from the HuggingFace model hub.
2. `prompt_engineering.ipynb`: How to use prompt engineering to improve the quality of the LLM responses.
3. `lora.ipynb`: Finetuning a model using QLoRA for a specific task.
4. `rag.ipynb`: How to talk with a PDF using Retrieval Augmented Generation (RAG).

The directory `llm_server` contains a simple API that can be used to serve a LLM in a remote machine.

### Usage

1. Go to a machine with a GPU (at least a 3090) and install the environment:

```shell
ssh remote_server
git clone https://github.com/RaulPPelaez/llm_handson
cd llm_handson
mamba create -n llm python==3.11 pip
mamba activate llm
pip install -r requirements.txt
```

2. Start a Jupyter Notebook server in a machine where you have a GPU:

Lets give it access to just one GPU.
```shell
CUDA_VISIBLE_DEVICES=0 jupyter notebook --no-browser --port 1234
```

3. Open the notebook in your laptop using a SSH tunnel:

```shell
	ssh -4 -NL 1234:localhost:1234 remote_server &
	firefox http://localhost:1234
```
