from fastapi import FastAPI
from transformers import (
    BertTokenizerFast,
)
import onnxruntime as rt
import time

app = FastAPI()

PRE_TRAINED_MODEL_NAME = "bert-base-cased"

sess = rt.InferenceSession("onnx_model.onnx")

sess.set_providers(["CUDAExecutionProvider"])
tokenizer = BertTokenizerFast.from_pretrained(PRE_TRAINED_MODEL_NAME)
BATCH_SIZE = 256
MAX_LEN = 60


@app.get("/")
async def root(data: str):
    start = time.time()

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
    encode = time.time()

    pred_onx = sess.run(
        None,
        {
            sess.get_inputs()[0].name: encoding["input_ids"],
            sess.get_inputs()[1].name: encoding["attention_mask"],
        },
    )

    onnx = time.time()

    return {
        data: str(pred_onx[0]),
        "time encode": "%.1f micros" % (1_000_000 * (encode - start)),
        "time onnx": "%.1f micros" % (1_000_000 * (onnx - encode)),
    }
