"""Minimal Metal compute backend for main.py.

Weights are uploaded as fp16 (default), int8 or int4 MTLBuffers depending on
METAL_QUANT; activations stay fp32. The shader source lives in shaders.metal and
is compiled once at import time through the Metal framework (no xcrun toolchain
needed).

    METAL_QUANT=fp16   exact; bandwidth-bound (~93-97 tok/s on an M4)
    METAL_QUANT=int8   per-group (64) absmax scales; ~150 tok/s, nearly
                       lossless (logit cosine 0.9995 vs fp16)
    METAL_QUANT=int4   per-group (32) absmax scales; ~185 tok/s but far too
                       lossy for a 0.5B model (logit cosine 0.75). Experimental.

Tuning knobs: METAL_CHUNK (layers per command buffer, default 4),
METAL_MV_TG (threads per quantized matvec threadgroup, default 256),
METAL_Q8_GROUP / METAL_Q4_GROUP (scale group sizes).

Public API used by main.py: array(), make_step(), argmax().
"""

import os
import struct
import sys

import numpy as np
import Metal

MAX_SEQ = 8192
TG = 256           # threads per threadgroup for reduction kernels
EPS = 1e-6         # rmsnorm epsilon, matches the numpy path
CHUNK = max(1, int(os.environ.get("METAL_CHUNK", "4")))
MV_TG = int(os.environ.get("METAL_MV_TG", "256"))
if MV_TG < 32 or MV_TG % 32 or MV_TG > 1024:
    raise ValueError("METAL_MV_TG must be a multiple of 32 in [32, 1024]")

KIND_F16, KIND_F32, KIND_I8, KIND_I4 = 0, 1, 2, 3
Q8_GROUP = int(os.environ.get("METAL_Q8_GROUP", "64"))
Q4_GROUP = int(os.environ.get("METAL_Q4_GROUP", "32"))
if Q8_GROUP <= 0 or Q4_GROUP <= 0 or Q4_GROUP % 8:
    raise ValueError("invalid quantization group size")

_quant = os.environ.get("METAL_QUANT", "").strip().lower()
if _quant in ("", "fp16", "f16", "16", "none"):
    QUANT = "fp16"
elif _quant in ("int8", "i8", "q8", "8"):
    QUANT = "int8"
elif _quant in ("int4", "i4", "q4", "4"):
    QUANT = "int4"
    print("warning: METAL_QUANT=int4 is very lossy on small models; "
          "use int8 unless you have checked the output quality", file=sys.stderr)
else:
    raise ValueError("METAL_QUANT must be fp16, int8 or int4 (got %r)" % _quant)

_device = Metal.MTLCreateSystemDefaultDevice()
if _device is None:
    raise RuntimeError("no Metal device available")
_queue = _device.newCommandQueue()

_SHADER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders.metal")
with open(_SHADER_PATH, "r", encoding="utf-8") as _fh:
    _source = _fh.read()

_library, _error = _device.newLibraryWithSource_options_error_(_source, None, None)
if _library is None:
    raise RuntimeError("Metal shader compilation failed: %s" % (_error,))

_pipelines = {}


def _pipeline(name):
    pso = _pipelines.get(name)
    if pso is not None:
        return pso
    fn = _library.newFunctionWithName_(name)
    if fn is None:
        raise RuntimeError("missing shader function: %s" % name)
    pso, err = _device.newComputePipelineStateWithFunction_error_(fn, None)
    if pso is None:
        raise RuntimeError("pipeline creation failed for %s: %s" % (name, err))
    _pipelines[name] = pso
    return pso


class Array:
    """A weight on the GPU.

    2D tensors are fp16 or quantized, 1D tensors (norms, biases) stay fp32.
    """

    __slots__ = ("buf", "shape", "kind", "host", "scales", "group")

    def __init__(self, buf, shape, kind, host=None, scales=None, group=0):
        self.buf = buf
        self.shape = tuple(shape)
        self.kind = kind
        self.host = host
        self.scales = scales
        self.group = group


def _alloc(nbytes):
    buf = _device.newBufferWithLength_options_(int(nbytes), 0)
    if buf is None:
        raise MemoryError("Metal buffer allocation failed (%d bytes)" % nbytes)
    return buf


def _upload(a, dtype, keep_host=False):
    a = np.ascontiguousarray(a, dtype=dtype)
    buf = _alloc(a.nbytes)
    view = np.frombuffer(buf.contents().as_buffer(a.nbytes), dtype=dtype)
    view[:] = a.reshape(-1)
    return buf, (a.copy() if keep_host else None)


def _float_buf(n):
    return _alloc(int(n) * 4)


def _quantize(a, bits):
    rows, cols = a.shape
    group = Q8_GROUP if bits == 8 else Q4_GROUP
    if cols % group:
        raise ValueError("quantized matvec needs column count divisible by %d" % group)
    groups = cols // group
    v = np.ascontiguousarray(a, dtype=np.float32).reshape(rows, groups, group)
    maxabs = np.abs(v).max(axis=2, keepdims=True)
    qmax = 127.0 if bits == 8 else 7.0
    scale = np.where(maxabs > 0.0, maxabs / qmax, 1.0).astype(np.float32)
    q = np.clip(np.rint(v / scale), -qmax, qmax).astype(np.int8)

    if bits == 8:
        data = q.reshape(rows, cols)
        kind = KIND_I8
        dtype = np.int8
    else:
        u = (q.astype(np.uint8) & 0x0F)
        pairs = u.reshape(rows, groups, group // 8, 8)
        packed = (pairs[:, :, :, :4] | (pairs[:, :, :, 4:] << 4)).astype(np.uint8)
        data = packed.reshape(rows, cols // 2)
        kind = KIND_I4
        dtype = np.uint8

    buf = _alloc(data.nbytes)
    np.frombuffer(buf.contents().as_buffer(data.nbytes), dtype=dtype)[:] = data.reshape(-1)
    sbuf, _ = _upload(scale.reshape(-1), np.float32)
    return Array(buf, (rows, cols), kind, scales=sbuf, group=group)


def array(a):
    a = np.asarray(a)
    if a.ndim >= 2:
        if QUANT == "int8":
            return _quantize(a, 8)
        if QUANT == "int4":
            return _quantize(a, 4)
        buf, _ = _upload(a, np.float16)
        return Array(buf, a.shape, KIND_F16)
    buf, host = _upload(a, np.float32, keep_host=True)
    return Array(buf, a.shape, KIND_F32, host=host)


_sizes = {}


def _size(n):
    s = _sizes.get(n)
    if s is None:
        s = _sizes[n] = Metal.MTLSizeMake(n, 1, 1)
    return s


def _encode(enc, name, buffers, consts):
    enc.setComputePipelineState_(_pipeline(name))
    for index, buf, offset in buffers:
        enc.setBuffer_offset_atIndex_(buf, offset, index)
    for index, fmt, value in consts:
        data = struct.pack("<" + fmt, value)
        enc.setBytes_length_atIndex_(data, len(data), index)


def _barrier(enc):
    # Metal hazard-tracks buffers within a command buffer; explicit barriers are
    # not needed and each one costs a Python->ObjC round trip.
    return


def _groups(enc, name, buffers, consts, groups, tg=TG):
    _encode(enc, name, buffers, consts)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(_size(groups), _size(tg))
    _barrier(enc)


def _threads(enc, name, buffers, consts, count):
    if count <= 0:
        return
    _encode(enc, name, buffers, consts)
    enc.dispatchThreads_threadsPerThreadgroup_(_size(count), _size(TG))
    _barrier(enc)


def _copy_half(enc, src, dst, elem_offset, n):
    _threads(enc, "copy_half",
             [(0, src, 0), (1, dst, 0)],
             [(2, "I", n), (3, "I", elem_offset)], n)


def _copy_uchar(enc, src, dst, offset, n):
    _threads(enc, "copy_uchar",
             [(0, src, 0), (1, dst, 0)],
             [(2, "I", n), (3, "I", offset)], n)


def _copy_float(enc, src, dst, elem_offset, n):
    _threads(enc, "copy_float",
             [(0, src, 0), (1, dst, 0)],
             [(2, "I", n), (3, "I", elem_offset)], n)


def _concat_rows(enc, arrays):
    """Concatenate same-width weight matrices row-wise."""
    rows = sum(a.shape[0] for a in arrays)
    cols = arrays[0].shape[1]
    kind = arrays[0].kind
    if any(a.kind != kind or a.shape[1] != cols for a in arrays):
        raise ValueError("cannot concatenate weights of differing layout")
    if kind == KIND_F16:
        buf = _alloc(rows * cols * 2)
        off = 0
        for a in arrays:
            _copy_half(enc, a.buf, buf, off, a.shape[0] * cols)
            off += a.shape[0] * cols
        return Array(buf, (rows, cols), KIND_F16)
    if kind in (KIND_I8, KIND_I4):
        rowbytes = cols if kind == KIND_I8 else cols // 2
        buf = _alloc(rows * rowbytes)
        off = 0
        for a in arrays:
            _copy_uchar(enc, a.buf, buf, off, a.shape[0] * rowbytes)
            off += a.shape[0] * rowbytes
        group = arrays[0].group
        groups = cols // group
        sbuf = _alloc(rows * groups * 4)
        off = 0
        for a in arrays:
            _copy_float(enc, a.scales, sbuf, off, a.shape[0] * groups)
            off += a.shape[0] * groups
        return Array(buf, (rows, cols), kind, scales=sbuf, group=group)
    raise ValueError("cannot concatenate fp32 weights")


def _concat_bias(parts, lengths):
    """Build one fp32 bias buffer; missing pieces become zeros."""
    if all(p is None for p in parts):
        return None
    out = np.zeros(sum(lengths), dtype=np.float32)
    off = 0
    for p, n in zip(parts, lengths):
        if p is not None:
            if len(p) != n:
                raise ValueError("bias length mismatch")
            out[off:off + n] = p
        off += n
    buf, host = _upload(out, np.float32, keep_host=True)
    return Array(buf, (len(out),), KIND_F32, host=host)


class _Runner:
    def __init__(self, W, n_layers, n_heads, n_kv, head, rope_base, rope_style):
        self.W = W
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.n_kv = int(n_kv)
        self.head = int(head)
        self.rope_base = float(rope_base)
        self.style = 0 if rope_style == "half" else 1

        if self.n_heads % self.n_kv:
            raise ValueError("n_heads must be a multiple of n_kv")

        emb = W["token_embd.weight"]
        self.emb = emb
        self.vocab, self.width = int(emb.shape[0]), int(emb.shape[1])
        self.ffn = int(W["blk.0.ffn_gate.weight"].shape[0])
        qsz = self.n_heads * self.head
        ksz = self.n_kv * self.head
        self.qkv_rows = qsz + 2 * ksz
        self.q_size = qsz
        self.k_size = ksz
        self.scale = self.head ** -0.5

        if self.width % 4 or self.ffn % 4 or qsz % 4:
            raise ValueError("matvec kernels require dimensions divisible by 4")

        self.x = _float_buf(self.width)
        self.h = _float_buf(self.width)
        self.qkv = _float_buf(self.qkv_rows)
        self.mixed = _float_buf(qsz)
        self.gateup = _float_buf(2 * self.ffn)
        self.logits = _float_buf(self.vocab)
        self.scores = _float_buf(self.n_heads * MAX_SEQ)

        self.emb_table = emb
        self.output_norm = W["output_norm.weight"].buf
        self.wout = W.get("output.weight", emb)

        self.layers = self._build_layers()

    def _build_layers(self):
        W = self.W
        cb = _queue.commandBuffer()
        enc = cb.computeCommandEncoder()
        layers = []
        for i in range(self.n_layers):
            p = "blk.%d." % i
            wqkv = _concat_rows(enc, [W[p + "attn_q.weight"],
                                      W[p + "attn_k.weight"],
                                      W[p + "attn_v.weight"]])
            qb = W.get(p + "attn_q.bias")
            kb = W.get(p + "attn_k.bias")
            vb = W.get(p + "attn_v.bias")
            bqkv = _concat_bias(
                [qb.host if qb is not None else None,
                 kb.host if kb is not None else None,
                 vb.host if vb is not None else None],
                [self.q_size, self.k_size, self.k_size])

            wgateup = _concat_rows(enc, [W[p + "ffn_gate.weight"],
                                         W[p + "ffn_up.weight"]])
            gb = W.get(p + "ffn_gate.bias")
            ub = W.get(p + "ffn_up.bias")
            bgateup = _concat_bias(
                [gb.host if gb is not None else None,
                 ub.host if ub is not None else None],
                [self.ffn, self.ffn])

            ob = W.get(p + "attn_output.bias")
            db = W.get(p + "ffn_down.bias")
            qn = W.get(p + "attn_q_norm.weight")
            kn = W.get(p + "attn_k_norm.weight")

            layers.append({
                "norm1": W[p + "attn_norm.weight"].buf,
                "norm2": W[p + "ffn_norm.weight"].buf,
                "wqkv": wqkv,
                "bqkv": bqkv.buf if bqkv is not None else None,
                "wo": W[p + "attn_output.weight"],
                "bo": ob.buf if ob is not None else None,
                "wgateup": wgateup,
                "bgateup": bgateup.buf if bgateup is not None else None,
                "wdown": W[p + "ffn_down.weight"],
                "bdown": db.buf if db is not None else None,
                "qnorm": qn.buf if qn is not None else None,
                "knorm": kn.buf if kn is not None else None,
                "keys": _float_buf(self.n_kv * MAX_SEQ * self.head),
                "values": _float_buf(self.n_kv * MAX_SEQ * self.head),
            })
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if cb.status() != 4:
            raise RuntimeError("Metal init failed: %s" % (cb.error(),))
        return layers

    def _rms(self, enc, xbuf, scale, ybuf, n, rows, xoff=0, yoff=0):
        _groups(enc, "rmsnorm",
                [(0, xbuf, xoff), (1, scale, 0), (2, ybuf, yoff)],
                [(3, "I", n), (4, "f", EPS)], rows)

    def _mv(self, enc, xbuf, w, ybuf, rows, cols, bias=None, residual=None):
        flags = (1 if bias is not None else 0) | (2 if residual is not None else 0)
        if w.kind == KIND_F16:
            _groups(enc, "matvec",
                    [(0, xbuf, 0), (1, w.buf, 0),
                     (2, bias if bias is not None else ybuf, 0),
                     (3, residual if residual is not None else ybuf, 0),
                     (4, ybuf, 0)],
                    [(5, "I", cols), (6, "I", flags)], rows)
        else:
            name = "matvec_i8" if w.kind == KIND_I8 else "matvec_i4"
            warps = MV_TG // 32
            groups = (rows + warps - 1) // warps
            _groups(enc, name,
                    [(0, xbuf, 0), (1, w.buf, 0), (2, w.scales, 0),
                     (3, bias if bias is not None else ybuf, 0),
                     (4, residual if residual is not None else ybuf, 0),
                     (5, ybuf, 0)],
                    [(6, "I", cols), (7, "I", flags), (8, "I", w.group),
                     (9, "I", rows)], groups, tg=MV_TG)

    def _embedding(self, enc, token):
        if self.emb.kind == KIND_F16:
            _threads(enc, "embedding_lookup",
                     [(0, self.emb.buf, 0), (1, self.x, 0)],
                     [(2, "I", self.width), (3, "I", token)], self.width)
        else:
            name = "embedding_lookup_i8" if self.emb.kind == KIND_I8 else "embedding_lookup_i4"
            _threads(enc, name,
                     [(0, self.emb.buf, 0), (1, self.emb.scales, 0),
                      (2, self.x, 0)],
                     [(3, "I", self.width), (4, "I", token),
                      (5, "I", self.emb.group)], self.width)

    def _layer(self, enc, i, pos, seq):
        L = self.layers[i]
        qbytes = self.q_size * 4
        kbytes = self.k_size * 4
        # attention block
        self._rms(enc, self.x, L["norm1"], self.h, self.width, 1)
        self._mv(enc, self.h, L["wqkv"], self.qkv,
                 self.qkv_rows, self.width, bias=L["bqkv"])
        if L["qnorm"] is not None:
            self._rms(enc, self.qkv, L["qnorm"], self.qkv,
                      self.head, self.n_heads)
            self._rms(enc, self.qkv, L["knorm"], self.qkv,
                      self.head, self.n_kv, xoff=qbytes, yoff=qbytes)
        _threads(enc, "rope",
                 [(0, self.qkv, 0), (1, self.qkv, qbytes)],
                 [(2, "I", self.n_heads), (3, "I", self.n_kv),
                  (4, "I", self.head), (5, "f", self.rope_base),
                  (6, "I", pos), (7, "I", self.style)],
                 (self.n_heads + self.n_kv) * (self.head // 2))
        _threads(enc, "kv_store",
                 [(0, self.qkv, qbytes), (1, self.qkv, qbytes + kbytes),
                  (2, L["keys"], 0), (3, L["values"], 0)],
                 [(4, "I", pos), (5, "I", self.n_kv),
                  (6, "I", self.head), (7, "I", MAX_SEQ)],
                 self.n_kv * self.head)
        _groups(enc, "attn_scores",
                [(0, self.qkv, 0), (1, L["keys"], 0), (2, self.scores, 0)],
                [(3, "I", seq), (4, "I", self.n_heads), (5, "I", self.n_kv),
                 (6, "I", self.head), (7, "I", MAX_SEQ), (8, "f", self.scale)],
                self.n_heads)
        _groups(enc, "attn_softmax",
                [(0, self.scores, 0)],
                [(1, "I", seq), (2, "I", MAX_SEQ)], self.n_heads)
        _threads(enc, "attn_out",
                 [(0, self.scores, 0), (1, L["values"], 0), (2, self.mixed, 0)],
                 [(3, "I", seq), (4, "I", self.n_heads), (5, "I", self.n_kv),
                  (6, "I", self.head), (7, "I", MAX_SEQ)],
                 self.q_size)
        self._mv(enc, self.mixed, L["wo"], self.x,
                 self.width, self.q_size, bias=L["bo"], residual=self.x)

        # feed-forward block
        self._rms(enc, self.x, L["norm2"], self.h, self.width, 1)
        self._mv(enc, self.h, L["wgateup"], self.gateup,
                 2 * self.ffn, self.width, bias=L["bgateup"])
        _threads(enc, "silu_mul",
                 [(0, self.gateup, 0), (1, self.gateup, self.ffn * 4)],
                 [(2, "I", self.ffn)], self.ffn)
        self._mv(enc, self.gateup, L["wdown"], self.x,
                 self.width, self.ffn, bias=L["bdown"], residual=self.x)

    def step(self, token, pos, cache=None):
        if pos < 0 or pos >= MAX_SEQ:
            raise ValueError("position %d outside Metal KV cache (max %d)" % (pos, MAX_SEQ))
        seq = pos + 1

        # Commit in chunks so the GPU starts running while later dispatches are
        # still being encoded on the CPU.
        cbs = []
        cb = _queue.commandBuffer()
        enc = cb.computeCommandEncoder()
        self._embedding(enc, int(token))
        for i in range(self.n_layers):
            self._layer(enc, i, pos, seq)
            if (i + 1) % CHUNK == 0 or (i + 1) == self.n_layers:
                enc.endEncoding()
                cb.commit()
                cbs.append(cb)
                cb = _queue.commandBuffer()
                enc = cb.computeCommandEncoder()

        self._rms(enc, self.x, self.output_norm, self.x, self.width, 1)
        self._mv(enc, self.x, self.wout, self.logits, self.vocab, self.width)
        enc.endEncoding()
        cb.commit()
        cbs.append(cb)

        cbs[-1].waitUntilCompleted()
        self.last_cbs = cbs
        for b in cbs:
            if b.status() != 4:
                raise RuntimeError("Metal command buffer failed: %s" % (b.error(),))

        view = np.frombuffer(self.logits.contents().as_buffer(self.vocab * 4),
                             dtype=np.float32)
        return view.copy()


def make_step(W, n_layers, n_heads, n_kv, head, rope_base, rope_style):
    return _Runner(W, n_layers, n_heads, n_kv, head, rope_base, rope_style).step


def argmax(scores):
    return int(np.argmax(scores))
