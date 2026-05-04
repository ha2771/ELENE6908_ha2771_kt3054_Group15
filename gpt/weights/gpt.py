#!/usr/bin/env python3
"""
pi_demo_distilgpt.py  -  Breathprint Edge inference demo on Raspberry Pi 5
============================================================================
Matches breathprint_distilgpt_final.ipynb / DistilGPT2 sensor classifier.

Runs only 10 test samples for each stage.

Files required in the same folder:
    pi_fp32.pt
    pi_int8.pt
    pi_pruned.pt
    pi_ee.pt
    distilgpt_sensor.onnx

Run:
    python pi_demo_distilgpt.py
"""

import os, sys, time, warnings, platform
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import GPT2Model, GPT2Config
    import onnxruntime as ort
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from ucimlrepo import fetch_ucirepo
except ImportError as e:
    sys.exit(
        f"Missing: {e}\n"
        "Run: pip install torch transformers onnxruntime scikit-learn ucimlrepo pandas numpy"
    )

# ── ARM / threading setup ─────────────────────────────────────────────────────
torch.set_num_threads(4)
_arch = platform.machine().lower()
_qbackend = "qnnpack" if _arch in ("aarch64", "arm64", "armv7l") else "fbgemm"
torch.backends.quantized.engine = _qbackend

N_SAMPLES   = 10
THRESHOLD   = 0.80
ORT_THREADS = 4

CLASS_MAP = {
    0: "Ethanol",
    1: "Ethylene",
    2: "Ammonia",
    3: "Acetaldehyde",
    4: "Acetone",
    5: "Toluene",
}

# ── Banner ────────────────────────────────────────────────────────────────────
print("=" * 66)
print("  Breathprint Edge - DistilGPT2 Pi 5 Demo")
print("=" * 66)
print(f"  Architecture     : {platform.machine()}")
print(f"  INT8 backend     : {_qbackend}  ({'ARM qnnpack' if _qbackend == 'qnnpack' else 'x86 fbgemm'})")
print(f"  ORT threads      : {ORT_THREADS}")
print(f"  Samples per model: {N_SAMPLES}")
print(f"  EE threshold     : {THRESHOLD}")
print()
print("  Optimization techniques to be run, in order:")
print("  ┌──────────────────────────────────────────────────────────┐")
print("  │ 1. FP32 Baseline — no compression/reference              │")
print("  │ 2. INT8 Dynamic PTQ — Linear weights quantized to int8    │")
print("  │ 3. ONNX Runtime — FP32 graph + CPU graph optimizations    │")
print("  │ 4. GPT2 Head Masking — selected heads zeroed/masked       │")
print("  │ 5. Early Exit — confidence-based layer skipping           │")
print("  └──────────────────────────────────────────────────────────┘")
print("=" * 66)


# =============================================================================
# MODEL DEFINITION — must match DistilGPT2 notebook
# =============================================================================
class DistilGPTSensorClassifier(nn.Module):
    N_LAYERS = 6
    D_MODEL  = 768

    def __init__(self, num_features=128, num_classes=6, dropout=0.1):
        super().__init__()
        D = self.D_MODEL
        self.num_features = num_features

        # Sensor scalar -> GPT token embedding
        self.value_proj = nn.Linear(1, D)

        # GPT is causal, so the READOUT token is placed at the END.
        self.readout_token = nn.Parameter(torch.randn(1, 1, D) * 0.02)

        # Instantiate DistilGPT2 architecture without needing internet/model download.
        # The trained weights are loaded from pi_*.pt files below.
        config = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_ctx=1024,
            n_embd=D,
            n_layer=self.N_LAYERS,
            n_head=12,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            use_cache=False,
        )
        self.distilgpt = GPT2Model(config)

        # Sensor data already comes as embeddings, so remove token embedding table.
        self.distilgpt.wte = nn.Identity()

        self.pre_classifier = nn.Linear(D, D)
        self.classifier     = nn.Linear(D, num_classes)
        self.dropout        = nn.Dropout(dropout)

        self.exit_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(D),
                nn.Linear(D, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )
            for _ in range(self.N_LAYERS)
        ])

    def _embed(self, x):
        """
        x: [B, 128]
        sequence: [sensor tokens] + [readout token]
        returns: [B, 129, 768]
        """
        x = self.value_proj(x.unsqueeze(-1))
        readout = self.readout_token.expand(x.size(0), -1, -1)
        hidden = torch.cat([x, readout], dim=1)

        seq_len = hidden.size(1)
        position_ids = torch.arange(seq_len, device=hidden.device).unsqueeze(0)
        pos_emb = self.distilgpt.wpe(position_ids)

        hidden = hidden + pos_emb
        hidden = self.distilgpt.drop(hidden)
        return hidden

    @staticmethod
    def _extract_hidden(block_out):
        # transformers versions differ: GPT2Block may return a Tensor, tuple,
        # or ModelOutput-like object. Never index a Tensor with [0], because
        # that removes the batch dimension and causes GPT2 attention to fail.
        if isinstance(block_out, torch.Tensor):
            return block_out
        if isinstance(block_out, (tuple, list)):
            return block_out[0]
        if hasattr(block_out, "last_hidden_state"):
            return block_out.last_hidden_state
        return block_out[0]

    def _run_transformer(self, emb):
        hidden = emb
        layer_hiddens = []

        for block in self.distilgpt.h:
            out = block(
                hidden,
                attention_mask=None,
                head_mask=None,
                use_cache=False,
                output_attentions=False,
            )
            hidden = self._extract_hidden(out)
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)
            layer_hiddens.append(hidden)

        hidden = self.distilgpt.ln_f(hidden)
        return hidden, layer_hiddens

    def forward(self, x, return_all_exits=False):
        emb = self._embed(x)
        hidden, layer_hiddens = self._run_transformer(emb)

        # Final readout token sees all previous sensor tokens.
        readout_hidden = hidden[:, -1]
        cls_out = self.dropout(F.relu(self.pre_classifier(readout_hidden)))
        final = self.classifier(cls_out)

        if not return_all_exits:
            return final

        exits = [self.exit_heads[i](h[:, -1]) for i, h in enumerate(layer_hiddens)]
        return exits, final


# =============================================================================
# DATA — same split as notebook
# =============================================================================
print("\nLoading UCI Gas Sensor dataset...")
gas = fetch_ucirepo(id=270)
X_raw = gas.data.features
y_raw = gas.data.targets

rows = []
for idx_str, row in X_raw.iterrows():
    class_id, concentration = idx_str.split(";")
    fv = {}
    for cell in row:
        if pd.isna(cell):
            continue
        i, v = str(cell).strip().split(":")
        fv[int(i)] = float(v)
    rows.append({"class": int(class_id), "concentration": float(concentration), **fv})

df = pd.DataFrame(rows)
df[1] = [float(str(c).strip().split(":")[1]) for c in y_raw.iloc[:, 0]]
feature_cols = sorted(c for c in df.columns if isinstance(c, int))
X = df[feature_cols].astype(float)
X.columns = [f"Feature{i}" for i in range(1, 129)]
y = df["class"].astype(int) - 1

X_tr, X_tmp, y_tr, y_tmp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
_, X_test, _, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
)

scaler = StandardScaler()
scaler.fit(X_tr)
X_test = scaler.transform(X_test)

X_demo = torch.tensor(X_test[:N_SAMPLES], dtype=torch.float32)
y_demo = y_test.values[:N_SAMPLES] if hasattr(y_test, "values") else y_test[:N_SAMPLES]
print(f"Using first {N_SAMPLES} test samples\n")


# =============================================================================
# HELPERS
# =============================================================================
def require_file(path):
    if not os.path.exists(path):
        sys.exit(f"ERROR: Required file not found: {path}")


def load_state_safely(model, path, strict=True):
    require_file(path)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=strict)
    model.eval()
    return model


def run_demo(stage_num, name, technique, what_it_does, fwd_fn):
    print(f"\n{'=' * 66}")
    print(f"  [{stage_num}/5] {name}")
    print(f"  Technique   : {technique}")
    print(f"  How it works: {what_it_does}")
    print(f"{'=' * 66}")

    correct, latencies = 0, []

    # Tiny warmup so first Python/PyTorch call does not dominate timing.
    with torch.no_grad():
        _ = fwd_fn(X_demo[0:1])

    for i in range(N_SAMPLES):
        x = X_demo[i:i + 1]
        t0 = time.perf_counter()
        with torch.no_grad():
            out = fwd_fn(x)
        ms = (time.perf_counter() - t0) * 1000

        pred = int(np.argmax(out) if isinstance(out, np.ndarray) else out.argmax(1).item())
        true = int(y_demo[i])
        correct += int(pred == true)
        latencies.append(ms)

        tick = "✓" if pred == true else "✗"
        print(
            f"  [{i + 1:02d}] {tick}  True: {CLASS_MAP[true]:14s}  "
            f"Pred: {CLASS_MAP[pred]:14s}  {ms:.1f} ms"
        )

    acc = correct / N_SAMPLES
    print(f"\n  Accuracy : {correct}/{N_SAMPLES}  ({100 * acc:.0f}%)")
    print(
        f"  Avg lat  : {np.mean(latencies):.1f} ms  "
        f"| Min: {np.min(latencies):.1f}  Max: {np.max(latencies):.1f} ms"
    )
    return np.mean(latencies), acc


results = {}


# =============================================================================
# STAGE 1 — BASELINE FP32
# =============================================================================
print("\n\nLoading Stage 1: FP32 baseline...")
m_fp32 = load_state_safely(DistilGPTSensorClassifier(), "pi_fp32.pt", strict=True)

lat, acc = run_demo(
    1,
    "Baseline FP32",
    "None — full float32",
    "All weights and activations are kept as 32-bit floats. This is the reference.",
    m_fp32,
)
results["FP32"] = (lat, acc)


# =============================================================================
# STAGE 2 — INT8 DYNAMIC PTQ
# =============================================================================
print(f"\n\nLoading Stage 2: INT8 Dynamic PTQ ({_qbackend} backend)...")
m_int8 = torch.quantization.quantize_dynamic(
    DistilGPTSensorClassifier(), {nn.Linear}, dtype=torch.qint8
)
m_int8 = load_state_safely(m_int8, "pi_int8.pt", strict=False)

lat, acc = run_demo(
    2,
    "INT8 Dynamic Post-Training Quantization",
    f"torch.quantization.quantize_dynamic  [backend: {_qbackend}]",
    "Linear layer weights are stored as int8. Activations remain float32 at runtime.",
    m_int8,
)
results["INT8"] = (lat, acc)


# =============================================================================
# STAGE 3 — ONNX RUNTIME
# =============================================================================
ONNX_PATH = "distilgpt_sensor.onnx"
print("\n\nLoading Stage 3: ONNX Runtime...")
require_file(ONNX_PATH)

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
so.intra_op_num_threads = ORT_THREADS
so.inter_op_num_threads = 1

sess = ort.InferenceSession(
    ONNX_PATH,
    sess_options=so,
    providers=["CPUExecutionProvider"],
)

# Detect the actual ONNX input name instead of assuming "input".
onnx_input_name = sess.get_inputs()[0].name

def ort_fwd(x):
    return sess.run(None, {onnx_input_name: x.detach().numpy().astype(np.float32)})[0]

lat, acc = run_demo(
    3,
    "ONNX Runtime",
    "FP32 weights exported to ONNX, executed with ORT CPUExecutionProvider",
    "ONNX Runtime applies graph optimizations, constant folding, and optimized CPU kernels.",
    ort_fwd,
)
results["ONNX"] = (lat, acc)


# =============================================================================
# STAGE 4 — GPT2 HEAD MASKING / STRUCTURED PRUNING EXPERIMENT
# =============================================================================
print("\n\nLoading Stage 4: GPT2 Head Masking...")
m_pruned = load_state_safely(DistilGPTSensorClassifier(), "pi_pruned.pt", strict=True)

lat, acc = run_demo(
    4,
    "GPT2 Head Masking / Structured Pruning",
    "Mask-based attention head pruning",
    "Low-importance attention heads were zeroed during training. Parameter count is unchanged, but selected heads do not contribute.",
    m_pruned,
)
results["Head Masking"] = (lat, acc)


# =============================================================================
# STAGE 5 — EARLY EXIT
# =============================================================================
print("\n\nLoading Stage 5: Early Exit...")
m_ee = load_state_safely(DistilGPTSensorClassifier(), "pi_ee.pt", strict=True)

@torch.no_grad()
def ee_single(m, x, thr):
    hidden = m._embed(x)

    for i, block in enumerate(m.distilgpt.h):
        out = block(
            hidden,
            attention_mask=None,
            head_mask=None,
            use_cache=False,
            output_attentions=False,
        )
        hidden = m._extract_hidden(out)
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)

        logits = m.exit_heads[i](hidden[:, -1])
        confidence = F.softmax(logits, dim=-1).max().item()

        if confidence >= thr:
            return logits, i

    hidden = m.distilgpt.ln_f(hidden)
    readout_hidden = hidden[:, -1]
    cls_out = m.dropout(F.relu(m.pre_classifier(readout_hidden)))
    logits = m.classifier(cls_out)
    return logits, m.N_LAYERS

print(f"\n{'=' * 66}")
print(f"  [5/5] Early Exit  (threshold={THRESHOLD})")
print("  Technique   : Confidence-based early exit")
print("  How it works: One lightweight exit head after each of 6 GPT2 blocks.")
print(f"                If max softmax probability >= {THRESHOLD}, stop and return.")
print("                Otherwise, run the full 6-layer DistilGPT2 pass.")
print(f"{'=' * 66}")

correct, latencies = 0, []
exit_counts = {i: 0 for i in range(m_ee.N_LAYERS + 1)}

# Warmup
_ = ee_single(m_ee, X_demo[0:1], THRESHOLD)

for i in range(N_SAMPLES):
    x = X_demo[i:i + 1]
    t0 = time.perf_counter()
    out, exit_layer = ee_single(m_ee, x, THRESHOLD)
    ms = (time.perf_counter() - t0) * 1000

    pred = int(out.argmax(-1).item())
    true = int(y_demo[i])
    correct += int(pred == true)
    latencies.append(ms)
    exit_counts[exit_layer] += 1

    tick = "✓" if pred == true else "✗"
    elabel = f"L{exit_layer}" if exit_layer < m_ee.N_LAYERS else "Full"
    print(
        f"  [{i + 1:02d}] {tick}  True: {CLASS_MAP[true]:14s}  "
        f"Pred: {CLASS_MAP[pred]:14s}  {ms:.1f} ms  [exit={elabel}]"
    )

acc = correct / N_SAMPLES
results["Early Exit"] = (np.mean(latencies), acc)
print(f"\n  Accuracy : {correct}/{N_SAMPLES}  ({100 * acc:.0f}%)")
print(f"  Avg lat  : {np.mean(latencies):.1f} ms")
print(f"  Exit distribution over {N_SAMPLES} samples:")
for layer in range(m_ee.N_LAYERS + 1):
    label = f"Layer {layer}" if layer < m_ee.N_LAYERS else "Full pass"
    cnt = exit_counts[layer]
    bar = "█" * cnt + "·" * (N_SAMPLES - cnt)
    print(f"    {label:10s}: {cnt:2d}/{N_SAMPLES}  [{bar}]")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
base_lat = results["FP32"][0]

print(f"\n\n{'=' * 66}")
print("  FINAL SUMMARY — Raspberry Pi 5 / CPU")
print(f"  {N_SAMPLES} samples | INT8 backend: {_qbackend}")
print(f"{'=' * 66}")
print(f"  {'Stage':<26} {'Avg Lat':>9} {'Acc':>6} {'Speedup':>8}")
print(f"  {'-' * 56}")

labels = {
    "FP32": "1. FP32 Baseline",
    "INT8": f"2. INT8 PTQ ({_qbackend})",
    "ONNX": "3. ONNX Runtime",
    "Head Masking": "4. GPT2 Head Masking",
    "Early Exit": "5. Early Exit",
}

for key, (lat, acc) in results.items():
    speedup = base_lat / lat
    faster = "faster" if speedup > 1 else "slower"
    print(
        f"  {labels[key]:<26} {lat:>8.1f}ms "
        f"{100 * acc:>5.0f}% {speedup:>7.2f}x  ({faster} than FP32)"
    )

print(f"{'=' * 66}")
print()
print("  Optimization summary:")
print("  Stage 1 — FP32 baseline/reference")
print("  Stage 2 — INT8 dynamic PTQ: Linear weights int8, activations float32")
print("  Stage 3 — ONNX Runtime: optimized CPU graph execution")
print("  Stage 4 — GPT2 head masking: heads zeroed, parameter count unchanged")
print("  Stage 5 — Early exit: skips later GPT2 blocks when confidence is high")
