import torch
import torch.nn.functional as F

# =========================
# LOAD
# =========================
data = torch.load("features.pt")

features = data["features"]
labels = data["labels"]

# =========================
# normalize
# =========================
features = F.normalize(features, dim=1)

# =========================
# GROUP BY SUBJECT
# =========================
subject_dict = {}

for feat, label in zip(features, labels):
    label = int(label)
    subject_dict.setdefault(label, []).append(feat)

# =========================
# DEBUG CHECK
# =========================
valid_subjects = {}

for subject, feats in subject_dict.items():
    if len(feats) >= 20:
        valid_subjects[subject] = feats[:20]

print("Total subjects:", len(subject_dict))
print("Valid subjects (>=20 images):", len(valid_subjects))

# =========================
# PAIRING
# =========================
genuine_scores = []
imposter_scores = []

subjects = list(valid_subjects.keys())

for subject in subjects:
    feats = valid_subjects[subject]

    gallery = feats[:10]
    probe = feats[10:]

    # ===== GENUINE =====
    for p in probe:
        for g in gallery:
            sim = F.cosine_similarity(p.unsqueeze(0), g.unsqueeze(0))
            genuine_scores.append(sim.item())

    # ===== IMPOSTER =====
    for other in subjects:
        if other == subject:
            continue

        other_gallery = valid_subjects[other][:10]

        for p in probe:
            for g in other_gallery:
                sim = F.cosine_similarity(p.unsqueeze(0), g.unsqueeze(0))
                imposter_scores.append(sim.item())

# =========================
# RESULTS
# =========================
print("Genuine pairs:", len(genuine_scores))
print("Imposter pairs:", len(imposter_scores))

if len(genuine_scores) > 0:
    print("Avg Genuine Similarity:", sum(genuine_scores)/len(genuine_scores))

if len(imposter_scores) > 0:
    print("Avg Imposter Similarity:", sum(imposter_scores)/len(imposter_scores))

# =========================
# SAVE
# =========================
torch.save({
    "genuine": genuine_scores,
    "imposter": imposter_scores
}, "similarity_scores.pt")

print("Saved similarity scores!")