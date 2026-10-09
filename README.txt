# inference

a very tiny and minimal inference engine capable of running llama2/llama3 & qwen2.5. supports gguf in form of fp32/fp16. quantized models are discarded for now. builtin support to offload tensor math to mlx and soon cuda.

$ wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-fp16.gguf
$ python3 main.py qwen2.5-0.5b-instruct-fp16.gguf "why is the sky blue?"
