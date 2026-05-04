#!/usr/bin/env python3
"""
pi_demo_distilbert.py  -  Breathprint Edge inference demo on Raspberry Pi 5
============================================================================
Matches breathprint_distilbert_final.ipynb exactly.

Files required (copy from Colab):
    pi_fp32.pt
    pi_int8.pt
    pi_pruned.pt
    pi_ee.pt
    distilbert_sensor.onnx

Run: python pi_demo_distilbert.py
"""

import os, sys, time, warnings, platform
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import DistilBertModel
    import onnxruntime as ort
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from ucimlrepo import fetch_ucirepo
except ImportError as e:
    sys.exit(f"Missing: {e}\nRun: pip install torch transformers onnxruntime scikit-learn ucimlrepo pandas")

# ── ARM / threading setup ─────────────────────────────────────────────────────
torch.set_num_threads(4)
_arch     = platform.machine().lower()
_qbackend = "qnnpack" if _arch in ("aarch64", "arm64", "armv7l") else "fbgemm"
torch.backends.quantized.engine = _qbackend

N_SAMPLES   = 10
THRESHOLD   = 0.80
ORT_THREADS = 4

CLASS_MAP = {0:"Ethanol", 1:"Ethylene", 2:"Ammonia",
             3:"Acetaldehyde", 4:"Acetone", 5:"Toluene"}

# ── Banner ────────────────────────────────────────────────────────────────────
print("=" * 62)
print("  Breathprint Edge - DistilBERT Pi 5 Demo")
print("=" * 62)
print(f"  Architecture     : {platform.machine()}")
print(f"  INT8 backend     : {_qbackend}  ({'ARM qnnpack' if _qbackend=='qnnpack' else 'x86 fbgemm'})")
print(f"  ORT threads      : {ORT_THREADS}")
print(f"  Samples per model: {N_SAMPLES}")
print(f"  EE threshold     : {THRESHOLD}")
print()
print("  Quantization techniques to be run, in order:")
print("  ┌─────────────────────────────────────────────────────┐")
print("  │ 1. FP32 Baseline — no quantization (reference)      │")
print("  │ 2. INT8 Dynamic PTQ — weights compressed to int8    │")
print("  │ 3. ONNX Runtime — FP32 + ARM NEON graph fusion      │")
print("  │ 4. Structured Pruning — 33% heads physically removed│")
print("  │ 5. Early Exit — confidence-based layer skipping     │")
print("  └─────────────────────────────────────────────────────┘")
print("=" * 62)


# =============================================================================
# MODEL DEFINITION  — must match breathprint_distilbert_final.ipynb
# =============================================================================
class DistilBertSensorClassifier(nn.Module):
    N_LAYERS = 6
    D_MODEL  = 768

    def __init__(self, num_features=128, num_classes=6, dropout=0.1):
        super().__init__()
        D = self.D_MODEL
        self.value_proj     = nn.Linear(1, D)
        self.feature_pos    = nn.Parameter(torch.randn(1, num_features + 1, D) * 0.02)
        self.cls_token      = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        self.distilbert     = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = nn.Linear(D, D)
        self.classifier     = nn.Linear(D, num_classes)
        self.dropout        = nn.Dropout(dropout)
        self.exit_heads     = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(D), nn.Linear(D, 256), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(256, num_classes),
            ) for _ in range(self.N_LAYERS)
        ])

    def _embed(self, x):
        x   = self.value_proj(x.unsqueeze(-1))
        cls = self.cls_token.expand(x.size(0), -1, -1)
        return torch.cat([cls, x], dim=1) + self.feature_pos

    def _run_transformer(self, emb):
        hidden, layer_hiddens = emb, []
        for layer in self.distilbert.transformer.layer:
            out    = layer(hidden, attn_mask=None, head_mask=None,
                           output_attentions=False)
            hidden = out[0] if isinstance(out, (tuple, list)) else out
            layer_hiddens.append(hidden)
        return hidden, layer_hiddens

    def forward(self, x, return_all_exits=False):
        emb    = self._embed(x)
        hidden, layer_hiddens = self._run_transformer(emb)
        cls_out = self.dropout(F.relu(self.pre_classifier(hidden[:, 0])))
        final   = self.classifier(cls_out)
        if not return_all_exits:
            return final
        exits = [self.exit_heads[i](h[:, 0]) for i, h in enumerate(layer_hiddens)]
        return exits, final


# =============================================================================
# HEAD PRUNING HELPERS  — V-norm scoring, matches final notebook
# =============================================================================
def get_heads_to_prune(m, n_prune=4):
    result = {}
    for layer_idx, layer in enumerate(m.distilbert.transformer.layer):
        attn = layer.attention
        n    = attn.n_heads
        d    = attn.dim // n
        norms = [attn.v_lin.weight.data[:, i*d:(i+1)*d].norm().item() for i in range(n)]
        result[layer_idx] = sorted(range(n), key=lambda i: norms[i])[:n_prune]
    return result

def physically_prune_heads(m, heads_to_prune):
    for layer_idx, prune_list in heads_to_prune.items():
        attn     = m.distilbert.transformer.layer[layer_idx].attention
        n_heads  = attn.n_heads
        head_dim = attn.dim // n_heads
        keep     = sorted(h for h in range(n_heads) if h not in prune_list)
        new_dim  = len(keep) * head_dim
        keep_idx = torch.cat([torch.arange(h*head_dim, (h+1)*head_dim) for h in keep])
        for proj_name in ["q_lin", "k_lin", "v_lin"]:
            old = getattr(attn, proj_name)
            new = nn.Linear(old.in_features, new_dim, bias=old.bias is not None)
            new.weight = nn.Parameter(old.weight[keep_idx].clone())
            if old.bias is not None:
                new.bias = nn.Parameter(old.bias[keep_idx].clone())
            setattr(attn, proj_name, new)
        old_o = attn.out_lin
        new_o = nn.Linear(new_dim, old_o.out_features, bias=old_o.bias is not None)
        new_o.weight = nn.Parameter(old_o.weight[:, keep_idx].clone())
        if old_o.bias is not None:
            new_o.bias = nn.Parameter(old_o.bias.clone())
        attn.out_lin = new_o
        attn.n_heads = len(keep)
        attn.dim     = new_dim


# =============================================================================
# DATA  — same split as notebook
# =============================================================================
print("\nLoading UCI Gas Sensor dataset...")
gas   = fetch_ucirepo(id=270)
X_raw = gas.data.features
y_raw = gas.data.targets

rows = []
for idx_str, row in X_raw.iterrows():
    class_id, concentration = idx_str.split(";")
    fv = {}
    for cell in row:
        if pd.isna(cell): continue
        i, v = str(cell).strip().split(":")
        fv[int(i)] = float(v)
    rows.append({"class": int(class_id), "concentration": float(concentration), **fv})

df = pd.DataFrame(rows)
df[1] = [float(str(c).strip().split(":")[1]) for c in y_raw.iloc[:, 0]]
feature_cols = sorted(c for c in df.columns if isinstance(c, int))
X = df[feature_cols].astype(float)
X.columns = [f"Feature{i}" for i in range(1, 129)]
y = df["class"].astype(int) - 1

X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
_, X_test, _, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

scaler = StandardScaler()
scaler.fit(X_tr)
X_test = scaler.transform(X_test)

X_demo = torch.tensor(X_test[:N_SAMPLES], dtype=torch.float32)
y_demo = y_test.values[:N_SAMPLES] if hasattr(y_test, "values") else y_test[:N_SAMPLES]
print(f"Using first {N_SAMPLES} test samples\n")


# =============================================================================
# INFERENCE HELPER
# =============================================================================
def run_demo(stage_num, name, technique, what_it_does, fwd_fn):
    print(f"\n{'='*62}")
    print(f"  [{stage_num}/5] {name}")
    print(f"  Technique   : {technique}")
    print(f"  How it works: {what_it_does}")
    print(f"{'='*62}")
    correct, latencies = 0, []
    for i in range(N_SAMPLES):
        x  = X_demo[i:i+1]
        t0 = time.perf_counter()
        with torch.no_grad():
            out = fwd_fn(x)
        ms = (time.perf_counter() - t0) * 1000
        pred     = int(np.argmax(out) if isinstance(out, np.ndarray) else out.argmax(1).item())
        true     = int(y_demo[i])
        correct += (pred == true)
        latencies.append(ms)
        tick = "✓" if pred == true else "✗"
        print(f"  [{i+1:02d}] {tick}  True: {CLASS_MAP[true]:14s}  "
              f"Pred: {CLASS_MAP[pred]:14s}  {ms:.1f} ms")
    acc = correct / N_SAMPLES
    print(f"\n  Accuracy : {correct}/{N_SAMPLES}  ({100*acc:.0f}%)")
    print(f"  Avg lat  : {np.mean(latencies):.1f} ms  "
          f"| Min: {np.min(latencies):.1f}  Max: {np.max(latencies):.1f} ms")
    return np.mean(latencies), acc

results = {}


# =============================================================================
# STAGE 1 — BASELINE FP32
# =============================================================================
print("\n\nLoading Stage 1: FP32 baseline...")
m_fp32 = DistilBertSensorClassifier()
m_fp32.load_state_dict(torch.load("pi_fp32.pt", map_location="cpu"))
m_fp32.eval()

lat, acc = run_demo(
    1, "Baseline FP32", "None — full float32",
    "All weights and activations kept as 32-bit floats. No compression.\n"
    "  This is the reference. Every other stage is compared against this.",
    m_fp32
)
results["FP32"] = (lat, acc)


# =============================================================================
# STAGE 2 — INT8 DYNAMIC PTQ
# =============================================================================
print(f"\n\nLoading Stage 2: INT8 PTQ ({_qbackend} backend)...")
m_int8 = torch.quantization.quantize_dynamic(
    DistilBertSensorClassifier(), {nn.Linear}, dtype=torch.qint8
)
m_int8.load_state_dict(torch.load("pi_int8.pt", map_location="cpu"), strict=False)
m_int8.eval()

lat, acc = run_demo(
    2, "INT8 Dynamic Post-Training Quantization",
    f"torch.quantization.quantize_dynamic  [backend: {_qbackend}]",
    "Weights of all nn.Linear layers compressed from float32 -> int8.\n"
    f"  Dequantized back to float32 at runtime before each matmul.\n"
    f"  No calibration data needed. ~2x smaller disk. Speedup depends on layer size.\n"
    f"  Backend '{_qbackend}' used because architecture is '{platform.machine()}'.",
    m_int8
)
results["INT8"] = (lat, acc)


# =============================================================================
# STAGE 3 — ONNX RUNTIME
# =============================================================================
ONNX_PATH = "distilbert_sensor.onnx"
print("\n\nLoading Stage 3: ONNX Runtime...")
if not os.path.exists(ONNX_PATH):
    sys.exit(f"ERROR: {ONNX_PATH} not found.")

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL
so.intra_op_num_threads     = ORT_THREADS
so.inter_op_num_threads     = 1
sess = ort.InferenceSession(ONNX_PATH, sess_options=so,
                             providers=["CPUExecutionProvider"])

def ort_fwd(x):
    return sess.run(None, {"input": x.detach().numpy().astype(np.float32)})[0]

lat, acc = run_demo(
    3, "ONNX Runtime",
    "FP32 weights exported to ONNX, executed via ORT CPUExecutionProvider",
    "Model graph exported to ONNX format (opset 18). ORT applies:\n"
    "  - Operator fusion (e.g. MatMul+Add merged)\n"
    "  - Constant folding and layout optimisation\n"
    "  - ARM NEON SIMD kernels on Pi 5 Cortex-A76\n"
    f"  Session: ORT_SEQUENTIAL | intra_threads={ORT_THREADS} | inter_threads=1",
    ort_fwd
)
results["ONNX"] = (lat, acc)


# =============================================================================
# STAGE 4 — STRUCTURED PRUNING
# =============================================================================
print("\n\nLoading Stage 4: Structured Pruning...")
m_pruned = DistilBertSensorClassifier()
heads_to_prune = get_heads_to_prune(m_pruned, n_prune=4)
physically_prune_heads(m_pruned, heads_to_prune)
m_pruned.load_state_dict(torch.load("pi_pruned.pt", map_location="cpu"))
m_pruned.eval()

before = 68_259_114   # full DistilBERT param count
after  = sum(p.numel() for p in m_pruned.parameters())

lat, acc = run_demo(
    4, "Structured Head Pruning",
    "Physical attention head removal (4 of 12 heads per layer, V-norm scoring)",
    f"Q/K/V/O weight matrices physically resized: 768x768 -> 768x512.\n"
    f"  Params: {before:,} -> {after:,}  ({(1-after/before)*100:.1f}% reduction).\n"
    f"  Heads selected by lowest L1 norm of V-projection slice.\n"
    f"  Fine-tuned 10 epochs post-pruning to recover accuracy.\n"
    f"  Unlike weight masking, this gives real FLOP and memory reduction.",
    m_pruned
)
results["Pruned"] = (lat, acc)


# =============================================================================
# STAGE 5 — EARLY EXIT
# =============================================================================
print("\n\nLoading Stage 5: Early Exit...")
m_ee = DistilBertSensorClassifier()
m_ee.load_state_dict(torch.load("pi_ee.pt", map_location="cpu"))
m_ee.eval()

@torch.no_grad()
def ee_single(m, x, thr):
    hidden = m._embed(x)
    for i, layer in enumerate(m.distilbert.transformer.layer):
        out    = layer(hidden, attn_mask=None, head_mask=None, output_attentions=False)
        hidden = out[0] if isinstance(out, (tuple, list)) else out
        logits = m.exit_heads[i](hidden[:, 0])
        if F.softmax(logits, dim=-1).max().item() >= thr:
            return logits, i
    cls = m.dropout(F.relu(m.pre_classifier(hidden[:, 0])))
    return m.classifier(cls), m.N_LAYERS

print(f"\n{'='*62}")
print(f"  [5/5] Early Exit  (threshold={THRESHOLD})")
print(f"  Technique   : Confidence-based early exit")
print(f"  How it works: One lightweight head after each of 6 transformer layers.")
print(f"                If max softmax probability >= {THRESHOLD}, stop and return.")
print(f"                Easy samples exit at layer 0 (skip 5/6 of transformer).")
print(f"                Hard samples fall through to the full 6-layer pass.")
print(f"{'='*62}")

correct, latencies = 0, []
exit_counts = {i: 0 for i in range(m_ee.N_LAYERS + 1)}

for i in range(N_SAMPLES):
    x  = X_demo[i:i+1]
    t0 = time.perf_counter()
    out, exit_layer = ee_single(m_ee, x, THRESHOLD)
    ms = (time.perf_counter() - t0) * 1000
    pred     = int(out.argmax(-1).item())
    true     = int(y_demo[i])
    correct += (pred == true)
    latencies.append(ms)
    exit_counts[exit_layer] += 1
    tick   = "✓" if pred == true else "✗"
    elabel = f"L{exit_layer}" if exit_layer < m_ee.N_LAYERS else "Full"
    print(f"  [{i+1:02d}] {tick}  True: {CLASS_MAP[true]:14s}  "
          f"Pred: {CLASS_MAP[pred]:14s}  {ms:.1f} ms  [exit={elabel}]")

acc = correct / N_SAMPLES
results["Early Exit"] = (np.mean(latencies), acc)
print(f"\n  Accuracy : {correct}/{N_SAMPLES}  ({100*acc:.0f}%)")
print(f"  Avg lat  : {np.mean(latencies):.1f} ms")
print(f"  Exit distribution over {N_SAMPLES} samples:")
for layer in range(m_ee.N_LAYERS + 1):
    label = f"Layer {layer}" if layer < m_ee.N_LAYERS else "Full pass"
    cnt   = exit_counts[layer]
    bar   = "█" * cnt + "·" * (N_SAMPLES - cnt)
    print(f"    {label:10s}: {cnt:2d}/{N_SAMPLES}  [{bar}]")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
base_lat = results["FP32"][0]

print(f"\n\n{'='*62}")
print(f"  FINAL SUMMARY — Raspberry Pi 5 ARM Cortex-A76")
print(f"  {N_SAMPLES} samples | INT8 backend: {_qbackend}")
print(f"{'='*62}")
print(f"  {'Stage':<22} {'Avg Lat':>9} {'Acc':>6} {'Speedup':>8}")
print(f"  {'-'*50}")

labels = {
    "FP32":       "1. FP32 Baseline",
    "INT8":       f"2. INT8 PTQ ({_qbackend})",
    "ONNX":       "3. ONNX Runtime",
    "Pruned":     "4. Struct. Pruning",
    "Early Exit": "5. Early Exit",
}
for key, (lat, acc) in results.items():
    speedup = base_lat / lat
    faster  = "faster" if speedup > 1 else "slower"
    print(f"  {labels[key]:<22} {lat:>8.1f}ms {100*acc:>5.0f}% {speedup:>7.2f}x  ({faster} than FP32)")

print(f"{'='*62}")
print()
print("  Quantization technique summary:")
print(f"  Stage 1 — No quantization (reference point)")
print(f"  Stage 2 — INT8 Dynamic PTQ: weights int8, activations float32")
print(f"            Backend auto-selected: {_qbackend} for {platform.machine()}")
print(f"  Stage 3 — ONNX Runtime: FP32 graph with ORT kernel optimisations")
print(f"  Stage 4 — Structural: 33% of attention heads physically removed")
print(f"  Stage 5 — Algorithmic: early exit skips transformer layers at runtime")
