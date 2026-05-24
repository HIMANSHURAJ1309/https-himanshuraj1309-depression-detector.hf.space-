import gradio as gr
import torch
import numpy as np
import os
from transformers import AutoTokenizer
from model import MentalRoBERTaPHQ9Model, build_phq9_mask, get_symptom_for_token

# ── CONFIG ──
LABEL_NAMES  = ["Not Depressed", "Mild", "Moderate", "Severe"]
LABEL_EMOJIS = ["✅", "🟡", "🟠", "🔴"]
LABEL_COLORS = ["#22c55e", "#facc15", "#f97316", "#ef4444"]
LABEL_BG     = ["#052e16", "#422006", "#431407", "#450a0a"]
MODEL_PATH   = "best_model.pt"
DEVICE       = torch.device("cpu")
HF_TOKEN     = os.environ.get("HF_TOKEN")

# ── LOAD MODEL ──
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("mental/mental-roberta-base", token=HF_TOKEN)
print("✅ Tokenizer loaded")

print("Loading model weights...")
model = MentalRoBERTaPHQ9Model(num_labels=4, dropout=0.3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("✅ Model ready")

def predict(text):
    text = text.strip()
    if not text or len(text) < 5:
        return ("","","","","")

    encoding = tokenizer(text, max_length=128, padding="max_length",
                         truncation=True, return_tensors="pt")
    input_ids      = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)
    tokens         = tokenizer.convert_ids_to_tokens(input_ids[0])
    phq9_mask      = build_phq9_mask(tokens).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits, attn_weights = model(input_ids, attention_mask, phq9_mask)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    pred_idx   = int(np.argmax(probs))
    confidence = float(probs[pred_idx]) * 100
    color      = LABEL_COLORS[pred_idx]
    bg         = LABEL_BG[pred_idx]
    emoji      = LABEL_EMOJIS[pred_idx]
    label      = LABEL_NAMES[pred_idx]

    # ── SEVERITY ──
    severity_html = f"""
    <div style='text-align:center;padding:28px 20px;border-radius:16px;
                background:{bg};border:2px solid {color};margin:8px 0;'>
        <div style='font-size:3em;margin-bottom:8px;'>{emoji}</div>
        <div style='font-size:2em;font-weight:800;color:{color};
                    letter-spacing:1px;'>{label}</div>
        <div style='font-size:1.1em;color:#e2e8f0;margin-top:8px;'>
            Confidence: <span style='color:{color};font-weight:700;'>{confidence:.1f}%</span>
        </div>
    </div>"""

    # ── PROBABILITY BARS ──
    bars_html = """<div style='padding:16px;background:#0f172a;border-radius:12px;
                               border:1px solid #1e293b;'>
        <div style='font-size:1em;font-weight:700;color:#e2e8f0;
                    margin-bottom:14px;letter-spacing:0.5px;'>
            📊 Class Probabilities
        </div>"""
    for i, (name, prob) in enumerate(zip(LABEL_NAMES, probs)):
        pct   = prob * 100
        c     = LABEL_COLORS[i]
        bold  = "font-weight:800;" if i == pred_idx else "font-weight:400;"
        alpha = "ff" if i == pred_idx else "99"
        bars_html += f"""
        <div style='margin-bottom:12px;'>
            <div style='display:flex;justify-content:space-between;
                        margin-bottom:5px;{bold}'>
                <span style='color:#e2e8f0;font-size:0.95em;'>
                    {LABEL_EMOJIS[i]} {name}
                </span>
                <span style='color:{c};font-size:0.95em;'>{pct:.1f}%</span>
            </div>
            <div style='background:#1e293b;border-radius:8px;height:10px;'>
                <div style='width:{pct}%;background:{c}{alpha};
                            height:10px;border-radius:8px;
                            transition:width 0.5s ease;'></div>
            </div>
        </div>"""
    bars_html += "</div>"

    # ── HEATMAP ──
    attn_np  = attn_weights[0].cpu().numpy()
    phq9_np  = phq9_mask[0].cpu().numpy()
    real_tok = [
        (tok, float(attn_np[i]), float(phq9_np[i]))
        for i, tok in enumerate(tokens)
        if tok not in ["<s>","</s>","<pad>"]
        and not tok.startswith("<")
        and attention_mask[0][i].item() == 1
    ]
    max_a = max([w for _,w,_ in real_tok]) if real_tok else 1.0

    heatmap_html = """
    <div style='padding:16px;background:#0f172a;border-radius:12px;
                border:1px solid #1e293b;'>
        <div style='font-size:1em;font-weight:700;color:#e2e8f0;margin-bottom:8px;'>
            🔥 PHQ-9 Attention Heatmap
        </div>
        <div style='font-size:0.82em;color:#94a3b8;margin-bottom:14px;'>
            🔴 Red border = PHQ-9 clinical keyword &nbsp;·&nbsp; Darker fill = higher attention weight
        </div>
        <div style='line-height:2.6;word-wrap:break-word;'>"""

    for tok, weight, is_phq9 in real_tok:
        clean = tok.replace("Ġ","").replace("##","").strip()
        if not clean:
            continue
        ratio = weight / max_a
        if is_phq9 == 1.0:
            bg_col  = f"rgba(239,68,68,{min(ratio+0.3,1):.2f})"
            border  = "2px solid #ef4444"
            txt_col = "#ffffff"
            title   = get_symptom_for_token(clean) or "PHQ-9 keyword"
        else:
            bg_col  = f"rgba(99,102,241,{ratio*0.6:.2f})"
            border  = "1px solid #334155"
            txt_col = "#e2e8f0"
            title   = f"Attention: {weight:.4f}"

        heatmap_html += f"""<span title='{title}'
            style='background:{bg_col};border:{border};color:{txt_col};
                   padding:4px 8px;border-radius:6px;margin:3px 2px;
                   display:inline-block;font-size:0.92em;font-weight:500;
                   cursor:default;'>{clean}</span>"""
    heatmap_html += "</div></div>"

    # ── SYMPTOMS ──
    symptoms = {}
    for tok, weight, is_phq9 in real_tok:
        if is_phq9 == 1.0:
            clean   = tok.replace("Ġ","").replace("##","")
            symptom = get_symptom_for_token(clean)
            if symptom:
                symptoms.setdefault(symptom, []).append(clean)

    if symptoms:
        sym_html = """<div style='padding:16px;background:#0f172a;border-radius:12px;
                                   border:1px solid #1e293b;'>
            <div style='font-size:1em;font-weight:700;color:#e2e8f0;margin-bottom:14px;'>
                🧠 PHQ-9 Symptoms Detected
            </div>"""
        for symptom, words in symptoms.items():
            sym_html += f"""
            <div style='margin-bottom:10px;padding:10px 14px;
                        background:#1e293b;border-left:4px solid #ef4444;
                        border-radius:6px;'>
                <span style='color:#fca5a5;font-weight:700;font-size:0.95em;'>
                    {symptom}
                </span>
                <span style='color:#94a3b8;margin-left:10px;font-size:0.9em;'>
                    → {", ".join(set(words))}
                </span>
            </div>"""
        sym_html += "</div>"
    else:
        sym_html = """<div style='padding:16px;background:#0f172a;border-radius:12px;
                                   border:1px solid #1e293b;color:#64748b;
                                   font-size:0.95em;'>
            No PHQ-9 clinical keywords detected in this text.
        </div>"""

    # ── ADVISORY ──
    advisories = [
        ("#052e16","#16a34a","#4ade80",
         "✅ No significant depressive indicators detected.",
         "This tool is for research purposes. If you have any concerns about your mental health, speaking to someone you trust is always a good step."),
        ("#422006","#d97706","#fcd34d",
         "🟡 Mild depressive indicators detected.",
         "Consider talking to a counsellor or trusted person. Maintaining routine, sleep, and social connection can help significantly."),
        ("#431407","#ea580c","#fdba74",
         "🟠 Moderate depressive indicators detected.",
         "We recommend consulting a mental health professional. A doctor or therapist can provide proper support and guidance."),
        ("#450a0a","#dc2626","#fca5a5",
         "🔴 Severe depressive indicators detected — please reach out for support.",
         """<b style='color:#fca5a5;'>India Crisis Helplines:</b><br>
         📞 iCall (TISS): <b>9152987821</b><br>
         📞 Vandrevala Foundation: <b>1860-2662-345</b> (24/7)<br>
         📞 NIMHANS Helpline: <b>080-46110007</b><br>
         📞 iCall WhatsApp: <b>9152987821</b>"""),
    ]
    bg_a, border_a, title_c, title, body = advisories[pred_idx]
    advisory_html = f"""
    <div style='padding:18px;background:{bg_a};border:2px solid {border_a};
                border-radius:12px;'>
        <div style='font-size:1.05em;font-weight:700;color:{title_c};
                    margin-bottom:10px;'>{title}</div>
        <div style='color:#e2e8f0;font-size:0.92em;line-height:1.7;'>{body}</div>
        <div style='margin-top:12px;font-size:0.8em;color:#94a3b8;'>
            ⚠️ This tool is a research prototype. It is NOT a substitute for professional clinical diagnosis.
        </div>
    </div>"""

    return severity_html, bars_html, heatmap_html, sym_html, advisory_html

SAMPLES = [
    "I feel so worthless and hopeless. Can't sleep at night, exhausted all day.",
    "Just feeling a bit down today, probably tired from work.",
    "I can't do this anymore. I just want to end it all. Nothing matters.",
    "Had a great day! Went for a walk and felt really good about myself.",
    "Can't focus on anything. Brain fog all the time. Feel like a failure.",
]

CSS = """
body, .gradio-container {
    background: #020617 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}
.gr-button-primary {
    background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    border: none !important; color: white !important;
    font-weight: 700 !important; font-size: 1.05em !important;
    border-radius: 10px !important; padding: 12px !important;
}
.gr-button-primary:hover {
    background: linear-gradient(135deg,#4f46e5,#7c3aed) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
}
.gr-button-secondary {
    background: #1e293b !important; border: 1px solid #334155 !important;
    color: #94a3b8 !important; border-radius: 10px !important;
}
.gr-textbox textarea, .gr-textbox input {
    background: #0f172a !important; color: #e2e8f0 !important;
    border: 1px solid #334155 !important; border-radius: 10px !important;
    font-size: 1em !important;
}
.gr-textbox label span {
    color: #94a3b8 !important; font-weight: 600 !important;
}
.gr-examples { background: #0f172a !important; border-radius:10px !important; }
.gr-examples table td { color: #94a3b8 !important; }
.gr-examples table tr:hover td { color: #e2e8f0 !important; }
.gr-accordion { background: #0f172a !important; border:1px solid #1e293b !important; }
.gr-accordion summary { color: #94a3b8 !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="Depression Severity Detector", css=CSS) as demo:

    gr.HTML("""
    <div style='text-align:center;padding:32px 20px 20px;'>
        <div style='font-size:2.8em;margin-bottom:10px;'>🧠</div>
        <h1 style='font-size:2.2em;font-weight:800;color:#e2e8f0;
                   margin:0 0 8px;letter-spacing:-0.5px;'>
            Depression Severity Detector
        </h1>
        <p style='color:#94a3b8;font-size:1em;margin:0 0 6px;'>
            MentalRoBERTa-BiLSTM · PHQ-9 Guided Attention · Dual-Layer Explainability
        </p>
        <p style='color:#64748b;font-size:0.85em;margin:0;'>
            M.Tech Research · Himanshu Raj [2403605] · BBAU Lucknow ·
            Supervisor: Prof. Sanjay K. Dwivedi
        </p>
    </div>""")

    gr.HTML("""
    <div style='max-width:860px;margin:0 auto 20px;padding:12px 18px;
                background:#1c1917;border:1px solid #78350f;border-radius:10px;
                color:#fbbf24;font-size:0.87em;'>
        ⚠️ <b>Research tool only.</b> Not a clinical diagnostic instrument.
        Do not use as a substitute for professional medical advice.
    </div>""")

    with gr.Column(elem_id="main-col",
                   scale=1):

        text_input = gr.Textbox(
            label="Enter a social media post or text",
            placeholder="Type or paste text here...",
            lines=4
        )
        with gr.Row():
            submit_btn = gr.Button("🔍  Analyse", variant="primary", scale=3)
            clear_btn  = gr.Button("✕  Clear", variant="secondary", scale=1)

        gr.Examples(examples=SAMPLES, inputs=text_input,
                    label="📝  Sample posts to try")

    gr.HTML("<div style='height:12px;'></div>")

    with gr.Row():
        severity_out = gr.HTML()
        bars_out     = gr.HTML()

    heatmap_out  = gr.HTML()

    with gr.Row():
        symptoms_out = gr.HTML()
        advisory_out = gr.HTML()

    with gr.Accordion("ℹ️  About this model", open=False):
        gr.HTML("""
        <div style='padding:16px;color:#94a3b8;font-size:0.9em;
                    line-height:1.8;background:#0f172a;border-radius:8px;'>
            <b style='color:#e2e8f0;'>Architecture:</b>
            MentalRoBERTa (768-dim) → Dropout(0.3) → BiLSTM (256×2=512-dim)
            → PHQ-9 Guided Attention (λ=0.495) → MLP (512→256→4)<br>
            <b style='color:#e2e8f0;'>Dataset:</b>
            DEPTWEET · 40,191 tweets · 4 severity classes<br>
            <b style='color:#e2e8f0;'>Results:</b>
            Accuracy 83.34% · Macro F1 0.6220 · Macro AUC 0.8686 · Severe F1 0.6431<br>
            <b style='color:#e2e8f0;'>Parameters:</b> 128,456,710<br>
            <b style='color:#e2e8f0;'>Novel contributions:</b>
            PHQ-9 Guided Attention · Dual-Layer XAI (LIME+SHAP) · 60% alignment score
        </div>""")

    gr.HTML("""
    <div style='text-align:center;padding:20px 0 10px;
                color:#475569;font-size:0.82em;border-top:1px solid #1e293b;
                margin-top:20px;'>
        Research only · Not for clinical use ·
        <a href='https://github.com/HIMANSHURAJ1309'
           style='color:#6366f1;text-decoration:none;'>GitHub</a> ·
        <a href='https://linkedin.com/in/himanshuraj1309'
           style='color:#6366f1;text-decoration:none;'>LinkedIn</a>
    </div>""")

    submit_btn.click(fn=predict, inputs=text_input,
                     outputs=[severity_out,bars_out,heatmap_out,symptoms_out,advisory_out])
    text_input.submit(fn=predict, inputs=text_input,
                      outputs=[severity_out,bars_out,heatmap_out,symptoms_out,advisory_out])
    def clear_all():
        return None, None, None, None, None

    def clear_all():
        return None, None, None, None, None

    clear_btn.click(fn=clear_all,
                    outputs=[severity_out,bars_out,heatmap_out,symptoms_out,advisory_out])
    clear_btn.click(fn=lambda: None, outputs=[text_input])

if __name__ == "__main__":
    demo.launch()
