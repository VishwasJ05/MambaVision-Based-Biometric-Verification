import torch
import torch.nn.functional as F

# =========================
# LOAD REAL + E FEATURES
# =========================
real = torch.load("features.pt")
e = torch.load("features_e.pt")

real_features = F.normalize(real["features"], dim=1)
real_labels = real["labels"]

e_features = F.normalize(e["features"], dim=1)
e_labels = e["labels"]

# =========================
# GROUP BY SUBJECT
# =========================
def group(features, labels):
    d = {}
    for f, l in zip(features, labels):
        l = int(l)
        d.setdefault(l, []).append(f)
    return d

real_dict = group(real_features, real_labels)
e_dict = group(e_features, e_labels)

# =========================
# ATTACK COMPARISON
# =========================
attack_scores = []

subjects = list(real_dict.keys())

for subject in subjects:
    if subject not in e_dict:
        continue

    real_gallery = real_dict[subject][:10]   # real 1–10
    e_probe = e_dict[subject][10:]          # E 11–20

    for p in e_probe:
        for g in real_gallery:
            sim = F.cosine_similarity(p.unsqueeze(0), g.unsqueeze(0))
            attack_scores.append(sim.item())

# =========================
# RESULTS
# =========================
print("Attack pairs:", len(attack_scores))
print("Avg Attack Similarity:", sum(attack_scores)/len(attack_scores))

torch.save({"attack": attack_scores}, "attack_scores.pt")
print("Saved attack_scores.pt")