# Demo BERT ONNX server written in rust

This demo showcase the use of onnxruntime-rs with a GPU on CUDA 11 served by actix-web and tokenized with Hugging Face tokenizer.

## Requirement

- Linux x86_64
- NVIDIA GPU with CUDA 11 (Not sure if CUDA 10 works)
- Rust (obviously)

## Installation

```bash
export ORT_USE_CUDA=1
cargo build --release
```

## Requirement

```bash
cargo run --release
```

or

```bash
export LD_LIBRARY_PATH=path/to/onnxruntime-linux-x64-gpu-1.8.0/lib:${LD_LIBRARY_PATH}
./target/release/onnx-server
```

## Call

```bash
curl http://localhost:8080/\?data=Hello+World
```

# Python alternative

To compare with standart python server with FastAPI, I've added the code for the same server in src called `python_alternative.py`

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
cd src
uvicorn python_alternative:app --reload --workers 1
```

## Call

```bash
curl http://localhost:8000/\?data=Hello+World
```
