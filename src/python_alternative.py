from fastapi import FastAPI
from transformers import (
    BertTokenizer,
)
import onnxruntime as rt

app = FastAPI()

PRE_TRAINED_MODEL_NAME = "bert-base-cased"

sess = rt.InferenceSession("onnx_model.onnx")

sess.set_providers(["CUDAExecutionProvider"])
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
BATCH_SIZE = 256
MAX_LEN = 60


@app.get("/")
async def root(data: str):
    encoding = tokenizer(
        [data],
        add_special_tokens=True,
        max_length=60,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="np",
    )
    pred_onx = sess.run(
        None,
        {
            sess.get_inputs()[0].name: encoding["input_ids"],
            sess.get_inputs()[1].name: encoding["attention_mask"],
        },
    )

    return {data: str(pred_onx[0])}
