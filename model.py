import os
import torch
import torch.nn as nn
from transformers import AutoModel

# ── PHQ-9 LEXICON ──
PHQ9_LEXICON = {
    "Q1_anhedonia": [
        "worthless","empty","no pleasure","lost interest","nothing matters",
        "numb","dead inside","feel nothing","no joy","meaningless","lifeless",
        "no point","cant enjoy","anhedonia"
    ],
    "Q2_low_mood": [
        "hopeless","sad","crying","depressed","miserable","down","darkness",
        "despair","heartbroken","broken","gloomy","cant go on","low mood",
        "unhappy","tearful","wept","weeping"
    ],
    "Q3_sleep": [
        "insomnia","cant sleep","sleeping too much","wide awake","nightmares",
        "up all night","restless night","no sleep","oversleeping","cant wake up",
        "sleep problems","awake all night","sleep deprived"
    ],
    "Q4_fatigue": [
        "tired","exhausted","no energy","drained","cant get up","weak",
        "worn out","no motivation","always tired","wiped out","fatigued",
        "lethargic","sluggish","no strength","depleted"
    ],
    "Q5_appetite": [
        "not eating","lost appetite","binge","no hunger","forgetting to eat",
        "cant eat","skipping meals","weight loss","overeating","eating too much",
        "no appetite","starving myself","food disgusts"
    ],
    "Q6_guilt": [
        "failure","shame","guilt","burden","useless","hate myself",
        "not good enough","let everyone down","self blame","regret",
        "worthless person","my fault","blame myself","ashamed","disgusted with myself"
    ],
    "Q7_concentration": [
        "cant focus","foggy","blank","cant think","confused","brain fog",
        "zoned out","cant study","forgetful","spacing out","cant concentrate",
        "mind blank","losing focus","distracted","cant remember"
    ],
    "Q8_psychomotor": [
        "slow","restless","agitated","frozen","cant move","sluggish",
        "paralyzed","everything takes effort","shaking","trembling",
        "moving slowly","feel stuck","body heavy","cant function"
    ],
    "Q9_suicidality": [
        "end it","want to die","not worth living","suicide","hurt myself",
        "give up","better off dead","no reason to live","suicidal",
        "kill myself","wish i was dead","cant take it anymore","end my life",
        "dont want to exist","disappear forever","self harm"
    ],
}

# ── FLAT KEYWORD SET ──
ALL_KEYWORDS = set()
for kws in PHQ9_LEXICON.values():
    ALL_KEYWORDS.update(kws)


def build_phq9_mask(tokens, max_length=128):
    """Build binary mask: 1 if token matches any PHQ-9 keyword."""
    mask = []
    for token in tokens[:max_length]:
        token_clean = (token.lower()
                       .replace("##", "")
                       .replace("\u0120", "")
                       .strip())
        hit = 0.0
        if len(token_clean) > 2:
            for kw in ALL_KEYWORDS:
                if kw in token_clean or token_clean in kw:
                    hit = 1.0
                    break
        mask.append(hit)
    mask = mask + [0.0] * (max_length - len(mask))
    return torch.tensor(mask[:max_length], dtype=torch.float)


def get_symptom_for_token(token):
    """Return the PHQ-9 symptom dimension for a token, or None."""
    token_clean = token.lower().replace("##", "").strip()
    for symptom, kws in PHQ9_LEXICON.items():
        for kw in kws:
            if kw in token_clean or token_clean in kw:
                return symptom.replace("_", " ").title()
    return None


# ── PHQ-9 ATTENTION LAYER ──
class PHQ9Attention(nn.Module):
    def __init__(self, hidden_size=512):
        super().__init__()
        self.attention    = nn.Linear(hidden_size, 1)
        self.lambda_phq9  = nn.Parameter(torch.tensor(0.5))

    def forward(self, bilstm_out, phq9_mask, attention_mask):
        scores = self.attention(bilstm_out).squeeze(-1)
        scores = scores + (self.lambda_phq9 * phq9_mask)
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        alpha  = torch.softmax(scores, dim=-1)
        context = torch.bmm(alpha.unsqueeze(1), bilstm_out).squeeze(1)
        return context, alpha


# ── FULL MODEL ──
class MentalRoBERTaPHQ9Model(nn.Module):
    def __init__(self, num_labels=4, dropout=0.3):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(
            "mental/mental-roberta-base",
            token=os.environ.get("HF_TOKEN")
        )
        self.bilstm = nn.LSTM(
            input_size=768, hidden_size=256, num_layers=2,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.phq9_attention = PHQ9Attention(hidden_size=512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, phq9_mask):
        out        = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        seq        = self.dropout(out.last_hidden_state)
        bilstm_out, _ = self.bilstm(seq)
        bilstm_out = self.dropout(bilstm_out)
        context, attn = self.phq9_attention(
            bilstm_out, phq9_mask, attention_mask
        )
        return self.classifier(context), attn
