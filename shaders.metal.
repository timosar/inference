// Metal compute kernels for a Llama/Qwen-style transformer.
// 2D weights live in half buffers, activations in float buffers.

#include <metal_stdlib>
using namespace metal;

inline float tg_sum(float v, threadgroup float* shared, uint tid, uint tgsize) {
    v = simd_sum(v);
    uint lane = tid & 31u;
    uint group = tid >> 5u;
    uint ngroups = (tgsize + 31u) >> 5u;
    if (lane == 0u) shared[group] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (group == 0u) {
        float x = (lane < ngroups) ? shared[lane] : 0.0f;
        x = simd_sum(x);
        if (lane == 0u) shared[0] = x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return shared[0];
}

inline float tg_max(float v, threadgroup float* shared, uint tid, uint tgsize) {
    v = simd_max(v);
    uint lane = tid & 31u;
    uint group = tid >> 5u;
    uint ngroups = (tgsize + 31u) >> 5u;
    if (lane == 0u) shared[group] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (group == 0u) {
        float x = (lane < ngroups) ? shared[lane] : -INFINITY;
        x = simd_max(x);
        if (lane == 0u) shared[0] = x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return shared[0];
}

kernel void embedding_lookup(device const half* table [[buffer(0)]],
                             device float* x [[buffer(1)]],
                             constant uint& width [[buffer(2)]],
                             constant uint& token [[buffer(3)]],
                             uint i [[thread_position_in_grid]]) {
    if (i < width) x[i] = float(table[(ulong)token * width + i]);
}

// One threadgroup per row. y = x / sqrt(mean(x^2) + eps) * scale
kernel void rmsnorm(device const float* x [[buffer(0)]],
                    device const float* scale [[buffer(1)]],
                    device float* y [[buffer(2)]],
                    constant uint& n [[buffer(3)]],
                    constant float& eps [[buffer(4)]],
                    uint tid [[thread_position_in_threadgroup]],
                    uint row [[threadgroup_position_in_grid]],
                    uint tgsize [[threads_per_threadgroup]]) {
    threadgroup float shared[32];
    device const float* xr = x + (ulong)row * n;
    device float* yr = y + (ulong)row * n;
    float acc = 0.0f;
    for (uint i = tid; i < n; i += tgsize) {
        float v = xr[i];
        acc += v * v;
    }
    float ss = tg_sum(acc, shared, tid, tgsize);
    float inv = rsqrt(ss / (float)n + eps);
    for (uint i = tid; i < n; i += tgsize) {
        yr[i] = xr[i] * inv * scale[i];
    }
}

// fp16 weights. One threadgroup per output row; flags bit 0 adds bias,
// bit 1 adds the residual.
kernel void matvec(device const float* x [[buffer(0)]],
                   device const half* w [[buffer(1)]],
                   device const float* bias [[buffer(2)]],
                   device const float* residual [[buffer(3)]],
                   device float* y [[buffer(4)]],
                   constant uint& cols [[buffer(5)]],
                   constant uint& flags [[buffer(6)]],
                   uint tid [[thread_position_in_threadgroup]],
                   uint row [[threadgroup_position_in_grid]],
                   uint tgsize [[threads_per_threadgroup]]) {
    threadgroup float shared[32];
    const device half4* w4 = (const device half4*)(w + (ulong)row * cols);
    const device float4* x4 = (const device float4*)x;
    uint c4 = cols >> 2;
    float acc = 0.0f;
    for (uint i = tid; i < c4; i += tgsize) {
        acc += dot(x4[i], float4(w4[i]));
    }
    float total = tg_sum(acc, shared, tid, tgsize);
    if (tid == 0u) {
        if (flags & 1u) total += bias[row];
        if (flags & 2u) total += residual[row];
        y[row] = total;
    }
}

// style 0: split-half (qwen / neox), style 1: interleaved (llama).
kernel void rope(device float* q [[buffer(0)]],
                 device float* k [[buffer(1)]],
                 constant uint& n_q [[buffer(2)]],
                 constant uint& n_kv [[buffer(3)]],
                 constant uint& head [[buffer(4)]],
                 constant float& base [[buffer(5)]],
                 constant uint& pos [[buffer(6)]],
                 constant uint& style [[buffer(7)]],
                 uint gid [[thread_position_in_grid]]) {
    uint halfd = head >> 1;
    uint total = (n_q + n_kv) * halfd;
    if (gid >= total) return;
    uint vec = gid / halfd;
    uint i = gid - vec * halfd;
    uint ia, ib;
    if (style == 0u) {
        ia = i;
        ib = i + halfd;
    } else {
        ia = 2u * i;
        ib = ia + 1u;
    }
    device float* v = (vec < n_q) ? (q + (ulong)vec * head)
                                  : (k + (ulong)(vec - n_q) * head);
    float freq = pow(base, -2.0f * (float)i / (float)head);
    float angle = (float)pos * freq;
    float c = cos(angle);
    float s = sin(angle);
    float a = v[ia];
    float b = v[ib];
    v[ia] = a * c - b * s;
    v[ib] = a * s + b * c;
}

kernel void kv_store(device const float* k [[buffer(0)]],
                     device const float* v [[buffer(1)]],
                     device float* keys [[buffer(2)]],
                     device float* values [[buffer(3)]],
                     constant uint& pos [[buffer(4)]],
                     constant uint& n_kv [[buffer(5)]],
                     constant uint& head [[buffer(6)]],
                     constant uint& max_seq [[buffer(7)]],
                     uint gid [[thread_position_in_grid]]) {
    uint total = n_kv * head;
    if (gid >= total) return;
    uint h = gid / head;
    uint d = gid - h * head;
    ulong idx = ((ulong)h * max_seq + pos) * head + d;
    keys[idx] = k[gid];
    values[idx] = v[gid];
}

// One threadgroup per query head.
kernel void attn_scores(device const float* q [[buffer(0)]],
                        device const float* keys [[buffer(1)]],
                        device float* scores [[buffer(2)]],
                        constant uint& seq [[buffer(3)]],
                        constant uint& n_heads [[buffer(4)]],
                        constant uint& n_kv [[buffer(5)]],
                        constant uint& head [[buffer(6)]],
                        constant uint& max_seq [[buffer(7)]],
                        constant float& scale [[buffer(8)]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint h [[threadgroup_position_in_grid]],
                        uint tgsize [[threads_per_threadgroup]]) {
    uint group = n_heads / n_kv;
    uint kh = h / group;
    device const float* qh = q + (ulong)h * head;
    device const float* kb = keys + (ulong)kh * max_seq * head;
    device float* sc = scores + (ulong)h * max_seq;
    for (uint s = tid; s < seq; s += tgsize) {
        device const float* kv = kb + (ulong)s * head;
        float acc = 0.0f;
        for (uint d = 0; d < head; d++) acc += qh[d] * kv[d];
        sc[s] = acc * scale;
    }
}

// One threadgroup per head.
kernel void attn_softmax(device float* scores [[buffer(0)]],
                         constant uint& seq [[buffer(1)]],
                         constant uint& max_seq [[buffer(2)]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint h [[threadgroup_position_in_grid]],
                         uint tgsize [[threads_per_threadgroup]]) {
    threadgroup float shared[32];
    device float* sc = scores + (ulong)h * max_seq;
    float m = -INFINITY;
    for (uint i = tid; i < seq; i += tgsize) m = max(m, sc[i]);
    m = tg_max(m, shared, tid, tgsize);
    float acc = 0.0f;
    for (uint i = tid; i < seq; i += tgsize) acc += exp(sc[i] - m);
    float sum = tg_sum(acc, shared, tid, tgsize);
    float inv = 1.0f / sum;
    for (uint i = tid; i < seq; i += tgsize) sc[i] = exp(sc[i] - m) * inv;
}

kernel void attn_out(device const float* probs [[buffer(0)]],
                     device const float* values [[buffer(1)]],
                     device float* out [[buffer(2)]],
                     constant uint& seq [[buffer(3)]],
                     constant uint& n_heads [[buffer(4)]],
                     constant uint& n_kv [[buffer(5)]],
                     constant uint& head [[buffer(6)]],
                     constant uint& max_seq [[buffer(7)]],
                     uint gid [[thread_position_in_grid]]) {
    uint total = n_heads * head;
    if (gid >= total) return;
    uint h = gid / head;
    uint d = gid - h * head;
    uint kh = h / (n_heads / n_kv);
    device const float* p = probs + (ulong)h * max_seq;
    device const float* vb = values + (ulong)kh * max_seq * head + d;
    float acc = 0.0f;
    for (uint s = 0; s < seq; s++) acc += p[s] * vb[(ulong)s * head];
    out[gid] = acc;
}

kernel void silu_mul(device float* gate [[buffer(0)]],
                     device const float* up [[buffer(1)]],
                     constant uint& n [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < n) {
        float g = gate[i];
        gate[i] = g / (1.0f + exp(-g)) * up[i];
    }
}

kernel void copy_half(device const half* src [[buffer(0)]],
                      device half* dst [[buffer(1)]],
                      constant uint& n [[buffer(2)]],
                      constant uint& offset [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i < n) dst[(ulong)offset + i] = src[i];
}

kernel void copy_uchar(device const uchar* src [[buffer(0)]],
                       device uchar* dst [[buffer(1)]],
                       constant uint& n [[buffer(2)]],
                       constant uint& offset [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    if (i < n) dst[(ulong)offset + i] = src[i];
}

kernel void copy_float(device const float* src [[buffer(0)]],
                       device float* dst [[buffer(1)]],
                       constant uint& n [[buffer(2)]],
                       constant uint& offset [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    if (i < n) dst[(ulong)offset + i] = src[i];
}

// --- 8-bit weights: one signed byte per element, one scale per `group` elements.

kernel void embedding_lookup_i8(device const char* table [[buffer(0)]],
                                device const float* scales [[buffer(1)]],
                                device float* x [[buffer(2)]],
                                constant uint& width [[buffer(3)]],
                                constant uint& token [[buffer(4)]],
                                constant uint& group [[buffer(5)]],
                                uint i [[thread_position_in_grid]]) {
    if (i < width) {
        uint ngroups = width / group;
        x[i] = float(table[(ulong)token * width + i]) *
               scales[(ulong)token * ngroups + i / group];
    }
}

kernel void matvec_i8(device const float* x [[buffer(0)]],
                      device const char* w [[buffer(1)]],
                      device const float* scales [[buffer(2)]],
                      device const float* bias [[buffer(3)]],
                      device const float* residual [[buffer(4)]],
                      device float* y [[buffer(5)]],
                      constant uint& cols [[buffer(6)]],
                      constant uint& flags [[buffer(7)]],
                      constant uint& group [[buffer(8)]],
                      constant uint& rows [[buffer(9)]],
                      uint tid [[thread_position_in_threadgroup]],
                      uint tg [[threadgroup_position_in_grid]],
                      uint tgsize [[threads_per_threadgroup]]) {
    uint lane = tid & 31u;
    uint row = tg * (tgsize >> 5u) + (tid >> 5u);
    if (row >= rows) return;
    const device char4* w4 = (const device char4*)(w + (ulong)row * cols);
    const device float4* x4 = (const device float4*)x;
    const device float* sc = scales + (ulong)row * (cols / group);
    uint c4 = cols >> 2;
    float acc = 0.0f;
    for (uint i = lane; i < c4; i += 32u) {
        acc += dot(x4[i], float4(w4[i])) * sc[(i * 4u) / group];
    }
    acc = simd_sum(acc);
    if (lane == 0u) {
        if (flags & 1u) acc += bias[row];
        if (flags & 2u) acc += residual[row];
        y[row] = acc;
    }
}

// --- 4-bit weights: two signed nibbles per byte (low nibble first),
//     one scale per `group` elements.

kernel void embedding_lookup_i4(device const uchar* table [[buffer(0)]],
                                device const float* scales [[buffer(1)]],
                                device float* x [[buffer(2)]],
                                constant uint& width [[buffer(3)]],
                                constant uint& token [[buffer(4)]],
                                constant uint& group [[buffer(5)]],
                                uint i [[thread_position_in_grid]]) {
    if (i < width) {
        // Each 4-byte word holds w0..w3 in its low nibbles and w4..w7 in its
        // high nibbles.
        uint pos = i & 7u;
        uchar b = table[(ulong)token * (width >> 1) + (i >> 3u) * 4u + (pos & 3u)];
        int v = (pos < 4u) ? int(b & 0x0Fu) : int(b >> 4);
        v = (v ^ 8) - 8;
        uint ngroups = width / group;
        x[i] = float(v) * scales[(ulong)token * ngroups + i / group];
    }
}

kernel void matvec_i4(device const float* x [[buffer(0)]],
                      device const uchar* w [[buffer(1)]],
                      device const float* scales [[buffer(2)]],
                      device const float* bias [[buffer(3)]],
                      device const float* residual [[buffer(4)]],
                      device float* y [[buffer(5)]],
                      constant uint& cols [[buffer(6)]],
                      constant uint& flags [[buffer(7)]],
                      constant uint& group [[buffer(8)]],
                      constant uint& rows [[buffer(9)]],
                      uint tid [[thread_position_in_threadgroup]],
                      uint tg [[threadgroup_position_in_grid]],
                      uint tgsize [[threads_per_threadgroup]]) {
    uint lane = tid & 31u;
    uint row = tg * (tgsize >> 5u) + (tid >> 5u);
    if (row >= rows) return;
    const device uint* w32 = (const device uint*)(w + (ulong)row * (cols >> 1));
    const device float4* x4 = (const device float4*)x;
    const device float* sc = scales + (ulong)row * (cols / group);
    uint c8 = cols >> 3;         // 8 weights per 32-bit word
    float acc = 0.0f;
    for (uint i = lane; i < c8; i += 32u) {
        uint ww = w32[i];
        uint lo = ww & 0x0F0F0F0Fu;
        uint hi = (ww >> 4) & 0x0F0F0F0Fu;
        float4 fl = float4(as_type<char4>(lo));
        float4 fh = float4(as_type<char4>(hi));
        fl = select(fl, fl - 16.0f, fl >= 8.0f);
        fh = select(fh, fh - 16.0f, fh >= 8.0f);
        float4 xa = x4[2u * i];
        float4 xb = x4[2u * i + 1u];
        acc += (dot(xa, fl) + dot(xb, fh)) * sc[(i * 8u) / group];
    }
    acc = simd_sum(acc);
    if (lane == 0u) {
        if (flags & 1u) acc += bias[row];
        if (flags & 2u) acc += residual[row];
        y[row] = acc;
    }
}
