import json
import numpy as np

def load_config(path="config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def score_risk(events, cfg):
    # Sum weighted counts; safety weights are negative
    w_d = cfg['danger_weights']
    w_s = cfg['safety_weights']
    s = 0.0
    for e in events:
        label = e['label']
        if label in w_d: s += w_d[label]
        if label in w_s: s += w_s[label]
    # logistic
    k = cfg['risk_logistic']['k']; x0 = cfg['risk_logistic']['x0']
    prob = 1.0/(1.0+np.exp(-k*(s - x0)))
    return float(max(0.0, min(1.0, prob)))

class EMA:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.value = None
    def update(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value
