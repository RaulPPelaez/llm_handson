import os
model_name=os.environ["HUGGINGFACE_MODEL"]
import huggingface_hub
huggingface_hub.snapshot_download(model_name)
