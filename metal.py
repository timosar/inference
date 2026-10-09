"""
tensor math offloaded to Apple Silicon METAL
"""

import mlx.core as mx

def array(x):
    return mx.array(x)

def argmax(scores):
    return int(mx.argmax(scores).item())

def make_step(W, n_layers, n_heads, n_kv, head, rope_base, rope_style):
    def norm(x, scale):
        return x / mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6) * scale

    def softmax(s):
        z = s - mx.max(s, axis=-1, keepdims=True)
        e = mx.exp(z)
        return e / mx.sum(e, axis=-1, keepdims=True)

    def rotate(x, pos):
        half = x.shape[-1] // 2
        freq = rope_base ** (-2 * mx.arange(half) / x.shape[-1])
        cos, sin = mx.cos(pos * freq), mx.sin(pos * freq)
        if rope_style == "half":
            a, b = x[..., :half], x[..., half:]
            return mx.concatenate((a * cos - b * sin, b * cos + a * sin), axis=-1)
        a, b = x[..., 0::2], x[..., 1::2]
        return mx.concatenate((a * cos - b * sin, a * sin + b * cos), axis=-1).reshape(x.shape)

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
            keys = mx.repeat(mx.stack(cache[i][0], axis=1), n_heads // n_kv, 0)
            vals = mx.repeat(mx.stack(cache[i][1], axis=1), n_heads // n_kv, 0)
            scores = mx.einsum("hd,hsd->hs", q, keys) / math_sqrt(head)
            mixed = mx.einsum("hs,hsd->hd", softmax(scores), vals).reshape(-1)
            x = x + linear(mixed, W[p + "attn_output.weight"])
            h = norm(x, W[p + "ffn_norm.weight"])
            gate = linear(h, W[p + "ffn_gate.weight"])
            up = linear(h, W[p + "ffn_up.weight"])
            x = x + linear(gate / (1 + mx.exp(-gate)) * up, W[p + "ffn_down.weight"])
        x = norm(x, W["output_norm.weight"])
        scores = linear(x, W.get("output.weight", W["token_embd.weight"]))
        mx.eval(scores)
        return scores

    return step

def math_sqrt(x):
    return mx.sqrt(mx.array(x))
