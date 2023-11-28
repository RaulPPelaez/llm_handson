# Server-client LLM API

This code is an example on how to create an API that serves an LLM for remote inference.

It uses [transformers](https://huggingface.co/docs/transformers/index) for dealing with the LLM, [fastapi](https://fastapi.tiangolo.com/) for the API and [uvicorn](https://www.uvicorn.org/) for serving the API.

The whole LLM inference server is ran through a docker image.

## Usage

The only requirement is to have docker installed with GPU access. The current model that is hardcoded in the Dockerfile is `openchat/openchat_3.5`, but you can change it to any other model from the [HuggingFace model hub](https://huggingface.co/models). It will take around 15GB of GPU memory.

1. Build the docker image:

```shell
docker build -t llm_server .
```

2. Run the docker image:

```shell
docker run --gpus 0 -p 8080:80 llm_server
```

3. Send a request to the API:

You can use a simple curl request:
```shell
curl -X POST http://localhost:8080/process \
     -H "Content-Type: application/json" \
     -d '{"prompt": "your_prompt_here", "max_new_tokens": "128", "temperature": "0.2"}'
```

Or you can use the provided python script:
```shell
python chat.py --prompt "your_prompt_here" --max_new_tokens 128 --temperature 0.2
```


## Notes

- The API is served in port 80 inside the docker container, but you can map it to any port you want in your host machine.  
- The API is served in the `/process` endpoint.  
- The API expects a JSON with the following format:  
```json
{
	"prompt": "your_prompt_here",
	"max_new_tokens": "128",
	"temperature": "0.2"
}
```
This API can be easily extended in `main.py`.  
- The API returns a JSON with the following format:  
```json
{
	"text": "model_response",
	"request_time": "0.1234"
}
```
- The docker image will download the LLM from the [HuggingFace model hub](https://huggingface.co/models) during build and store the weights inside the image. Instead, you can download the weights to your local machine and mount them to the docker image during runtime.  
To do so, first download the weights:  
```shell
pip install huggingface_hub
#The model name must be the same as the one in the Dockerfile
export HUGGINGFACE_MODEL="openchat/openchat_3.5"
python download.py
```

Then, run the docker image with the mounted weights:  
```shell
docker run --gpus 0 -p 8080:80 -v ~/.cache/huggingface:/root/.cache/huggingface llm_server
```

- If you want to play around with the generation code you can mount the current directory to the docker image when running it, this way the image will use the code in the current directory instead of the one in the image when starting:  
```shell
docker run --gpus 0 -p 8080:80 -v $(pwd):/workdir/ llm_server
```

