import os
import logging
import time
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# FastAPI code
# These lines are the only thing required to make this a FastAPI app.
# Then, each function can be an entry point to the API by adding the @app decorator.
# The type annotations are used to inform FastAPI of the expected input and output types.
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class ProcessRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.2


class ProcessResponse(BaseModel):
    text: str
    request_time: float


# The model and tokenizer are loaded once when the app starts and are kept in global memory.
model_name = os.environ["HUGGINGFACE_MODEL"]
logger.info("Loading model: {}".format(model_name))
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # This is just to avoid a warning
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="cuda:0"
)
model.eval()


@torch.no_grad()
def generate(
    prompt,
    model,
    tokenizer,
    **kwargs,
):
    """Generate a response to a prompt.

    Args:
        prompt (str): The prompt to generate a response to.
        model (transformers.PreTrainedModel): The model to use for generation.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for generation.
        **kwargs: Additional keyword arguments to pass to model.generate.

    Returns:
        torch.Tensor: The generated tokens.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    ).to("cuda")
    inputs = inputs.to(model.device)
    question_length = len(inputs[0])
    logger.info("Sending {} tokens to model".format(question_length))
    logger.info(f"Using the following generator kwargs: {kwargs}")
    output = model.generate(
        inputs,
        return_dict_in_generate=True,
        output_scores=True,
        renormalize_logits=True,
        do_sample=True,
        **kwargs,
    )
    output_tokens = output.sequences[0][question_length:]
    # Top 5 most probable first tokens:
    probable_tokens = torch.topk(
        torch.nn.functional.softmax(output.scores[0][0], dim=-1), 5
    )
    tokens_with_probs = [
        (tokenizer.decode(tok), prob)
        for tok, prob in zip(probable_tokens.indices, probable_tokens.values)
    ]
    tokens_with_probs = "\n".join(
        [f"'\033[91m{token}\033[00m' with probability {prob}" for token, prob in tokens_with_probs]
    )
    logger.info(f"\033[1m Top 5 most probable first tokens:\033[00m\n{tokens_with_probs}")
    generated_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
    return generated_text


@app.post("/process")
async def process_request(input_data: ProcessRequest) -> ProcessResponse:
    """Process a request to the API.

    Args:
        input_data (ProcessRequest): The request to the API.

    Returns:
        ProcessResponse: The response from the API.
    """
    logger.info("Using device: {}".format(model.device))
    print(
        "\033[92m {}\033[00m".format(
            f"Prompt: {input_data.prompt[:20]}...{input_data.prompt[-20:]}"
        )
    )
    t0 = time.perf_counter()
    prompt = input_data.prompt
    del input_data.prompt
    generated_text = generate(
        prompt,
        model,
        tokenizer,
        **input_data.dict(),
    )
    reqtime = time.perf_counter() - t0
    logger.info(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
    print("\033[91m {}\033[00m".format(f"Generated text: {generated_text}"))
    return ProcessResponse(text=generated_text, request_time=reqtime)
