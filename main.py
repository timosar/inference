import sys, struct, math, platform
import numpy as np
import metal
import time

"""

inference engine for llama / qwen architecture
assumes FP32/FP16 model. quantized models are discarded for now.
builtin support for apple silicon metal (MLX) offloading (tensor math including matmul, rmsnorm, softmax, rope)

1. parse given gguf model (extract header, metadata & tensors)
2. locate the start of tensors and their weights
3. load the weights into memory
4. read the architecture settings from metadata
5. tokenize the prompt
6. lookup every token in the prompt in token_embed.weight
7. run it through rmsnorm, rope, residual add and feed-forward block
8. pick next token with argmax
9. interate through step()
10. decode new ids back to text
"""

if len(sys.argv) < 2:
    print("usage: python3.11 main.py <model.gguf> 'prompt'")
    sys.exit()
else:
    path, prompt = sys.argv[1], sys.argv[2]
    f = open(path, "rb")

def read(fmt):
    n = struct.calcsize("<" + fmt)
    data = f.read(n)
    if len(data) != n:
        raise EOFError("file ended early")
    return struct.unpack("<" + fmt, data)[0]

def read_string():
    return f.read(read("Q")).decode(errors="replace")

def read_value(kind):
    if kind == 8:
        return read_string()
    if kind == 9:
        item, count = read("I"), read("Q")
        return [read_value(item) for _ in range(count)]
    return read(["B", "b", "H", "h", "I", "i", "f", "?", "s", "a", "Q", "q", "d"][kind])

# --- header ---
if f.read(4) != b"GGUF":
    raise ValueError("not a GGUF file")
_, n_tensors, n_meta = read("I"), read("Q"), read("Q")

info = {read_string(): read_value(read("I")) for _ in range(n_meta)}
tensors = []
for _ in range(n_tensors):
    name = read_string() 
    rank = read("I")
    """
    if rank == 1:
        print("found tensor of type FP16\n")
    elif rank == 0:
        print("found tensor of type FP32\n")
    """
    tensors.append((name, [read("Q") for _ in range(rank)], read("I"), read("Q")))

align = info.get("general.alignment", 32)
data_start = (f.tell() + align - 1) & ~(align - 1)

def load(shape, storage, offset):
    if storage not in (0, 1):
        raise ValueError("only F32/F16 weights")
    dtype = "<f4" if storage == 0 else "<f2"
    f.seek(data_start + offset)
    raw = f.read(math.prod(shape) * np.dtype(dtype).itemsize)
    return np.frombuffer(raw, dtype=dtype).reshape(shape[::-1]).astype(np.float32)

W = {name: load(*rest) for name, *rest in tensors}
f.close()

# --- model shape. Qwen uses the same layer names as Llama ---
arch = info.get("general.architecture", "llama")
if arch not in ("llama", "qwen2", "qwen3"):
    raise ValueError(f"unsupported architecture: {arch}")

def setting(name, default=None):
    return info.get(f"{arch}.{name}", default)

n_layers = setting("block_count")
n_heads = setting("attention.head_count")
n_kv = setting("attention.head_count_kv", n_heads)
width = setting("embedding_length")
head = setting("attention.key_length", width // n_heads)
rope_base = setting("rope.freq_base", 10000.0)
eos = info.get("tokenizer.ggml.eos_token_id", 2)
rope_style = "half" if arch.startswith("qwen") else "pair"

apple = sys.platform == "darwin" and platform.machine() == "arm64"
if apple:
    W = {name: metal.array(weight) for name, weight in W.items()}
    step = metal.make_step(W, n_layers, n_heads, n_kv, head, rope_base, rope_style)
    take = metal.argmax
else:
    def norm(x, scale):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6) * scale

    def softmax(s):
        z = s - s.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    def rotate(x, pos):
        half = x.shape[-1] // 2
        freq = rope_base ** (-2 * np.arange(half) / x.shape[-1])
        cos, sin = np.cos(pos * freq), np.sin(pos * freq)
        if rope_style == "half":
            a, b = x[..., :half], x[..., half:]
            return np.concatenate((a * cos - b * sin, b * cos + a * sin), axis=-1)
        a, b = x[..., 0::2], x[..., 1::2]
        y = np.empty_like(x)
        y[..., 0::2], y[..., 1::2] = a * cos - b * sin, a * sin + b * cos
        return y

    def linear(x, weight, bias=None):
        y = x @ weight.T
        return y if bias is None else y + bias

    def step(token, pos, cache):
        x = W["token_embd.weight"][token]
        for i in range(n_layers):
            p = f"blk.{i}."
            h = norm(x, W[p + "attn_norm.weight"])
            q = linear(h, W[p + "attn_q.weight"], W.get(p + "attn_q.bias")).reshape(n_heads, head)
            k = linear(h, W[p + "attn_k.weight"], W.get(p + "attn_k.bias")).reshape(n_kv, head)
            v = linear(h, W[p + "attn_v.weight"], W.get(p + "attn_v.bias")).reshape(n_kv, head)
            if p + "attn_q_norm.weight" in W:
                q, k = norm(q, W[p + "attn_q_norm.weight"]), norm(k, W[p + "attn_k_norm.weight"])
            q, k = rotate(q, pos), rotate(k, pos)
            cache[i][0].append(k)
            cache[i][1].append(v)
            keys = np.repeat(np.stack(cache[i][0], 1), n_heads // n_kv, 0)
            vals = np.repeat(np.stack(cache[i][1], 1), n_heads // n_kv, 0)
            scores = np.einsum("hd,hsd->hs", q, keys) / math.sqrt(head)
            mixed = np.einsum("hs,hsd->hd", softmax(scores), vals).reshape(-1)
            x = x + linear(mixed, W[p + "attn_output.weight"])
            h = norm(x, W[p + "ffn_norm.weight"])
            gate = linear(h, W[p + "ffn_gate.weight"])
            up = linear(h, W[p + "ffn_up.weight"])
            x = x + linear(gate / (1 + np.exp(-gate)) * up, W[p + "ffn_down.weight"])
        x = norm(x, W["output_norm.weight"])
        return linear(x, W.get("output.weight", W["token_embd.weight"]))

    take = lambda scores: int(scores.argmax())

# --- tokenizer ---
# Qwen token 0 is "!". A space inside a token is U+0120, written Ġ.
vocab = info["tokenizer.ggml.tokens"]
merges = info.get("tokenizer.ggml.merges", [])
rank = {tuple(pair.split(" ", 1)): i for i, pair in enumerate(merges)}

bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
cs, extra = bs[:], 0
for b in range(256):
    if b not in bs:
        bs.append(b)
        cs.append(256 + extra)
        extra += 1
byte_map = dict(zip(bs, map(chr, cs)))
byte_unmap = {ch: b for b, ch in byte_map.items()}
lookup = {token: i for i, token in enumerate(vocab)}

def bpe(piece):
    parts = list(piece)
    while len(parts) > 1:
        pairs = [(rank.get((a, b), 10**9), i) for i, (a, b) in enumerate(zip(parts, parts[1:]))]
        score, i = min(pairs)
        if score == 10**9:
            break
        parts[i:i + 2] = [parts[i] + parts[i + 1]]
    return parts

def encode(text):
    if arch.startswith("qwen"):
        text = "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n"
    ids = []
    for i, chunk in enumerate(text.split(" ")):
        if i:
            chunk = "\u0120" + chunk
        raw = "".join(byte_map[b] for b in chunk.encode())
        ids.extend(lookup[part] for part in bpe(raw))
    return ids

def decode(ids):
    text = "".join(vocab[i] for i in ids).replace("\u0120", " ")
    out = bytearray()
    for ch in text:
        out.append(byte_unmap[ch] if ch in byte_unmap else ord(ch) if ord(ch) < 128 else 32)
    return out.decode("utf-8", errors="replace")

# --- generate. Only print new tokens, not the chat wrapper ---
ids = encode(prompt)
cache = [[[], []] for _ in range(n_layers)]
for pos, token in enumerate(ids):
    scores = step(token, pos, cache)




out = ""
n_tokens = 0
start = time.perf_counter()
for n in range(64):
    token = take(scores)
    piece = decode([token])
    if token == eos or "<|im_end|>" in out + piece or "<|endoftext|>" in out + piece:
        break
    out += piece
    n_tokens += 1
    print(piece, end="", flush=True)
    scores = step(token, len(ids) + n, cache)
elapsed = time.perf_counter() - start
print()
print(f"{n_tokens / elapsed:.1f} tok/s ({n_tokens} tokens in {elapsed:.2f}s)")

