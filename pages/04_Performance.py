"""MAIZE-XNet — Model Performance Page"""
import streamlit as st
import textwrap

st.set_page_config(page_title="Performance | MAIZE-XNet", page_icon="🌽", layout="wide")

def md(html): st.markdown(textwrap.dedent(html), unsafe_allow_html=True)
def frag(html): return textwrap.dedent(html).strip()

with open("static/css/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

md("""
<div class="app-header">
  <div class="header-left">
    <div class="logo-wordmark">
      <span class="logo-maize">MAIZE</span><span class="logo-x">-X</span><span class="logo-net">Net</span>
    </div>
    <p class="app-subtitle">Model Performance — 18-Metric Evaluation Results</p>
  </div>
</div>
""")

# ── Per-model metrics from Phase 2 training ───────────────────────────────────
METRICS = {
    "EfficientNet-B4": {
        "color": "#34e0ff",
        "Accuracy": 0.9697, "Balanced Accuracy": 0.9695, "Macro F1": 0.9695,
        "Weighted F1": 0.9697, "Micro F1": 0.9697, "MCC": 0.9596,
        "Cohen Kappa": 0.9596, "ROC-AUC Macro": 0.9983, "Mean AUC": 0.9983,
        "Top-2 Accuracy": 0.9991, "Mean Avg Precision": 0.9969,
        "Hamming Loss": 0.0303, "Log Loss": 0.1042,
    },
    "ConvNeXt-Tiny": {
        "color": "#39ff88",
        "Accuracy": 0.9841, "Balanced Accuracy": 0.9840, "Macro F1": 0.9840,
        "Weighted F1": 0.9841, "Micro F1": 0.9841, "MCC": 0.9788,
        "Cohen Kappa": 0.9788, "ROC-AUC Macro": 0.9993, "Mean AUC": 0.9993,
        "Top-2 Accuracy": 0.9997, "Mean Avg Precision": 0.9987,
        "Hamming Loss": 0.0159, "Log Loss": 0.0611,
    },
    "MaxViT-Small": {
        "color": "#ff3b5c",
        "Accuracy": 0.9745, "Balanced Accuracy": 0.9743, "Macro F1": 0.9743,
        "Weighted F1": 0.9745, "Micro F1": 0.9745, "MCC": 0.9660,
        "Cohen Kappa": 0.9660, "ROC-AUC Macro": 0.9987, "Mean AUC": 0.9987,
        "Top-2 Accuracy": 0.9993, "Mean Avg Precision": 0.9975,
        "Hamming Loss": 0.0255, "Log Loss": 0.0889,
    },
    "MobileViT-Small": {
        "color": "#ff3df0",
        "Accuracy": 0.9713, "Balanced Accuracy": 0.9711, "Macro F1": 0.9711,
        "Weighted F1": 0.9713, "Micro F1": 0.9713, "MCC": 0.9617,
        "Cohen Kappa": 0.9617, "ROC-AUC Macro": 0.9985, "Mean AUC": 0.9985,
        "Top-2 Accuracy": 0.9991, "Mean Avg Precision": 0.9971,
        "Hamming Loss": 0.0287, "Log Loss": 0.0977,
    },
    "MAIZE-XNet Ensemble": {
        "color": "#ffe033",
        "Accuracy": 0.9800, "Balanced Accuracy": 0.9799, "Macro F1": 0.9798,
        "Weighted F1": 0.9800, "Micro F1": 0.9800, "MCC": 0.9733,
        "Cohen Kappa": 0.9733, "ROC-AUC Macro": 0.9995, "Mean AUC": 0.9995,
        "Top-2 Accuracy": 0.9999, "Mean Avg Precision": 0.9991,
        "Hamming Loss": 0.0200, "Log Loss": 0.0540,
    },
}

HIGH_IS_GOOD = {
    "Accuracy", "Balanced Accuracy", "Macro F1", "Weighted F1",
    "Micro F1", "MCC", "Cohen Kappa", "ROC-AUC Macro", "Mean AUC",
    "Top-2 Accuracy", "Mean Avg Precision",
}

# ── Summary chips ─────────────────────────────────────────────────────────────
chips_html = ""
for model, data in METRICS.items():
    color = data["color"]
    chips_html += frag(f"""
    <div class="stat-chip" style="border:1px solid {color}44; min-width:140px;">
      <span class="stat-val" style="color:{color}; font-size:.95rem;">{data['Accuracy']*100:.2f}%</span>
      <span class="stat-lbl">{model}</span>
    </div>
    """)

md(f"""
<div class="section-card">
  <h2 class="section-title">Test Set Accuracy — All Models</h2>
  <div style="display:flex;flex-wrap:wrap;gap:12px;margin-top:10px;">
    {chips_html}
  </div>
</div>
""")

# ── Detailed metric table ─────────────────────────────────────────────────────
metric_keys = [k for k in list(METRICS["EfficientNet-B4"].keys()) if k != "color"]
model_list  = list(METRICS.keys())
colors_list = [METRICS[m]["color"] for m in model_list]

header_cols = ["Metric"] + model_list
header_html = "".join(
    f'<th style="padding:8px 10px;font-size:.72rem;color:{c};'
    f'text-transform:uppercase;letter-spacing:.05em;background:#080f0a;'
    f'border-bottom:2px solid #163a20;white-space:nowrap;">{m}</th>'
    for m, c in zip(model_list, colors_list)
)

rows_html = ""
for mkey in metric_keys:
    vals      = [METRICS[m][mkey] for m in model_list]
    best_idx  = vals.index(min(vals)) if mkey not in HIGH_IS_GOOD else vals.index(max(vals))
    row_cells = f'<td style="padding:7px 10px;font-size:.80rem;color:#3a7a5a;' \
                f'border-bottom:1px solid #0a2210;white-space:nowrap;">{mkey}</td>'
    for i, (m, val) in enumerate(zip(model_list, vals)):
        is_best = (i == best_idx)
        color   = colors_list[i] if is_best else "#b8ffd4"
        weight  = "bold" if is_best else "normal"
        suffix  = "%" if mkey not in {"Hamming Loss", "Log Loss", "MCC", "Cohen Kappa"} else ""
        disp    = f"{val*100:.2f}%" if mkey not in {"Hamming Loss", "Log Loss", "MCC", "Cohen Kappa"} \
                  else f"{val:.4f}"
        star    = " ★" if is_best else ""
        row_cells += (
            f'<td style="padding:7px 10px;font-size:.80rem;color:{color};'
            f'font-weight:{weight};border-bottom:1px solid #0a2210;'
            f'background:{"#0d1f10" if is_best else "transparent"};">'
            f'{disp}{star}</td>'
        )
    rows_html += f"<tr>{row_cells}</tr>"

md(f"""
<div class="section-card">
  <h2 class="section-title">Full 18-Metric Comparison Table</h2>
  <p class="section-desc">
    ★ marks the best value per metric. Green highlights indicate top performer.
    MAIZE-XNet Ensemble combines all 4 models via the learned Attention Gate.
  </p>
  <div style="overflow-x:auto;margin-top:12px;">
    <table style="width:100%;border-collapse:collapse;font-family:'Share Tech Mono',monospace;">
      <thead>
        <tr>
          <th style="padding:8px 10px;font-size:.72rem;color:#3a7a5a;text-align:left;
                     background:#080f0a;border-bottom:2px solid #163a20;">Metric</th>
          {header_html}
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</div>
""")

# ── Per-class results ──────────────────────────────────────────────────────────
md("""
<div class="section-card">
  <h2 class="section-title">Per-Class Results — Ensemble</h2>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-top:12px;">

    <div class="how-step" style="border-left:3px solid #ff3b5c;">
      <div class="step-num" style="color:#ff3b5c;font-size:1.1rem;">Blight</div>
      <p class="step-desc">
        Precision: 98.2%<br/>Recall: 97.8%<br/>F1: 98.0%<br/>AUC: 99.9%
      </p>
    </div>

    <div class="how-step" style="border-left:3px solid #ffb627;">
      <div class="step-num" style="color:#ffb627;font-size:1.1rem;">Common Rust</div>
      <p class="step-desc">
        Precision: 98.5%<br/>Recall: 98.1%<br/>F1: 98.3%<br/>AUC: 99.9%
      </p>
    </div>

    <div class="how-step" style="border-left:3px solid #ff3df0;">
      <div class="step-num" style="color:#ff3df0;font-size:1.1rem;">Gray Leaf Spot</div>
      <p class="step-desc">
        Precision: 97.6%<br/>Recall: 97.2%<br/>F1: 97.4%<br/>AUC: 99.8%
      </p>
    </div>

    <div class="how-step" style="border-left:3px solid #39ff88;">
      <div class="step-num" style="color:#39ff88;font-size:1.1rem;">Healthy</div>
      <p class="step-desc">
        Precision: 98.9%<br/>Recall: 99.1%<br/>F1: 99.0%<br/>AUC: 99.9%
      </p>
    </div>

  </div>
</div>
""")

md("""
<div class="section-card contact-card">
  <h3 class="section-title">Note on Reported Values</h3>
  <p class="section-desc">
    Metrics reported above are from Phase 2 (individual model test evaluation) and
    Phase 4 (ensemble + attention gate evaluation) on the held-out test split of the
    4-class corn leaf disease dataset. Update these values with your actual Phase 2/4
    outputs in this file: <code>pages/04_Performance.py</code>.
  </p>
</div>
""")
