"""Audit of the dataset and the evaluation of the selected model (TF-IDF + Linear SVM, C = 5).

Run from the project folder:   .venv/bin/python audit/ml_audit.py
Writes audit/audit_report.md, audit/audit_metrics.json and figures in audit/.

Checks: split reproduction, exact and near-duplicate leakage across splits, dataset artefacts that act as
label shortcuts, full metric set on the held-out test set, probability calibration, and two
generalisation tests (time-based split, topic/subject shift). It never edits the notebooks or models.
"""
import json, re, sys, time
from pathlib import Path

import joblib, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "audit"
REAL_C, FAKE_C = "#2A6FB5", "#D97706"


def clean(t):                                       # identical to Experiment 3
    t = t.lower()
    t = re.sub(r"reuters|http\S+|<.*?>", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return " ".join(t.split())


def metrics(y, pred, score):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "accuracy": accuracy_score(y, pred), "precision_fake": precision_score(y, pred, zero_division=0),
        "recall_fake": recall_score(y, pred, zero_division=0), "f1_fake": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, score), "pr_auc": average_precision_score(y, score),
        "false_positive_rate": fp / (fp + tn), "false_negative_rate": fn / (fn + tp),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def ece(y, prob, bins=10):
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob >= lo) & (prob < hi) if hi < 1 else (prob >= lo) & (prob <= hi)
        if m.any():
            total += m.mean() * abs(prob[m].mean() - y[m].mean())
    return total


def fit_eval(train_text, y_train, test_text, y_test):
    vec = TfidfVectorizer(max_features=50000, stop_words="english")
    m = LinearSVC(C=5).fit(vec.fit_transform(train_text), y_train)
    X = vec.transform(test_text)
    return metrics(y_test, m.predict(X), m.decision_function(X))


def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def main():
    t0 = time.time()
    R = {}

    # ---------- 1. reproduce the Experiment 3 split, keeping metadata ----------
    fake = pd.read_csv(ROOT / "Fake.csv"); fake["label"] = 1
    true = pd.read_csv(ROOT / "True.csv"); true["label"] = 0
    raw = pd.concat([true, fake], ignore_index=True)
    df = raw[raw["text"].str.strip() != ""].drop_duplicates(subset=["title", "text"]).reset_index(drop=True)
    X = (df["title"] + " " + df["text"]).apply(clean)
    y = df["label"]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)
    saved_te = pd.read_csv(ROOT / "test.csv")
    R["split"] = {
        "raw_articles": len(raw), "after_cleaning": len(df), "train": len(X_tr), "val": len(X_va), "test": len(X_te),
        "matches_saved_test_csv": bool((saved_te["text"].fillna("").values == X_te.fillna("").values).all()),
    }
    print("split reproduced:", R["split"])

    # ---------- 2. duplicates and near-duplicates across splits ----------
    tr_set, va_set = set(X_tr), set(X_va)
    R["exact_duplicates"] = {
        "test_in_train": int(sum(t in tr_set for t in X_te)), "val_in_train": int(sum(t in tr_set for t in X_va)),
        "test_in_val": int(sum(t in va_set for t in X_te)),
        "empty_after_cleaning": int((X.str.strip() == "").sum()),
        "duplicate_cleaned_texts_overall": int(X.duplicated().sum()),
    }
    tfidf = joblib.load(ROOT / "tfidf.joblib")
    A, B = tfidf.transform(X_te), tfidf.transform(X_tr)
    max_sim = np.zeros(A.shape[0])
    for i in range(0, A.shape[0], 400):                    # chunked cosine similarity (rows are L2-normalised)
        max_sim[i:i + 400] = (A[i:i + 400] @ B.T).max(axis=1).toarray().ravel()
    R["near_duplicates_test_vs_train"] = {f">= {t}": int((max_sim >= t).sum()) for t in (0.99, 0.95, 0.9, 0.8)}
    print("duplicates:", R["exact_duplicates"], R["near_duplicates_test_vs_train"])

    # ---------- 3. the selected model on the held-out test set ----------
    pipe = joblib.load(ROOT / "final_pipeline.joblib")
    te_mask = X_te.str.strip() != ""
    Xt, yt = X_te[te_mask], y_te[te_mask].values
    score = pipe.decision_function(Xt); pred = (score > 0).astype(int)
    R["test_metrics"] = metrics(yt, pred, score)
    keep = max_sim[te_mask.values] < 0.9
    R["test_metrics_without_near_duplicates"] = metrics(yt[keep], pred[keep], score[keep])
    print("test:", {k: fmt(v) for k, v in R["test_metrics"].items() if k != "confusion"})

    # ---------- 4. dataset artefacts that act as label shortcuts ----------
    full = (df["title"] + " " + df["text"])
    art = {
        "contains '(Reuters)'": full.str.contains(r"\(Reuters\)", regex=True),
        "starts with an all-caps dateline": df["text"].str.match(r"^\s*[A-Z][A-Z .,/'-]{2,40}\("),
        "contains 'featured image'": full.str.contains("featured image", case=False),
        "contains 'getty images'": full.str.contains("getty images", case=False),
        "contains 'pic.twitter.com'": full.str.contains("pic.twitter.com", case=False),
        "contains 'said'": full.str.contains(r"\bsaid\b", case=False, regex=True),
        "title is mostly upper-case words": df["title"].apply(lambda s: np.mean([w.isupper() for w in s.split()] or [0]) > 0.5),
    }
    R["artefacts_percent_of_class"] = {k: {"real": round(100 * v[y == 0].mean(), 1), "fake": round(100 * v[y == 1].mean(), 1)}
                                       for k, v in art.items()}
    R["subjects"] = {"real": sorted(df.loc[y == 0, "subject"].unique()), "fake": sorted(df.loc[y == 1, "subject"].unique())}
    R["subjects_overlap"] = sorted(set(R["subjects"]["real"]) & set(R["subjects"]["fake"]))
    dates = pd.to_datetime(df["date"].str.strip(), format="mixed", errors="coerce")
    R["date_range"] = {lbl: [str(dates[y == v].min().date()), str(dates[y == v].max().date())] for lbl, v in (("real", 0), ("fake", 1))}
    R["unparseable_dates"] = int(dates.isna().sum())
    # shortcut probe: how well do the first 12 cleaned words alone predict the label?
    first = X.str.split().str[:12].str.join(" ")
    R["shortcut_first_12_words"] = fit_eval(first[X_tr.index], y_tr, first[X_te.index], y_te.values)
    # ablation: drop the word "said" at test time
    nosaid = Xt.str.replace(r"\bsaid\b", " ", regex=True)
    s2 = pipe.decision_function(nosaid)
    R["ablation_without_said"] = metrics(yt, (s2 > 0).astype(int), s2)
    print("artefacts done")

    # ---------- 5. calibration ----------
    Xtr_v = tfidf.transform(X_tr[X_tr.str.strip() != ""]); ytr_v = y_tr[X_tr.str.strip() != ""]
    cal = CalibratedClassifierCV(LinearSVC(C=5), method="sigmoid", cv=5).fit(Xtr_v, ytr_v)
    prob_cal = cal.predict_proba(tfidf.transform(Xt))[:, 1]
    prob_ui = 1 / (1 + np.exp(-2 * score))                  # the "lean" the UI used to draw the bar
    R["calibration"] = {
        "ui_sigmoid_2x_margin": {"brier": brier_score_loss(yt, prob_ui), "ece": ece(yt, prob_ui)},
        "platt_calibrated_cv5": {"brier": brier_score_loss(yt, prob_cal), "ece": ece(yt, prob_cal)},
    }
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for p, name, c in ((prob_ui, "UI lean = sigmoid(2 x margin)", FAKE_C), (prob_cal, "Platt-calibrated (5-fold)", REAL_C)):
        fy, fx = calibration_curve(yt, p, n_bins=10, strategy="quantile")
        ax[0].plot(fx, fy, "o-", color=c, label=name)
    ax[0].plot([0, 1], [0, 1], "--", color="#999"); ax[0].set_title("Reliability diagram (test)", loc="left")
    ax[0].set_xlabel("Predicted P(fake)"); ax[0].set_ylabel("Observed fraction fake"); ax[0].legend(frameon=False, fontsize=8)
    pr, rc, _ = precision_recall_curve(yt, score)
    ax[1].plot(rc, pr, color=FAKE_C); ax[1].set_title(f"Precision-recall (AP {R['test_metrics']['pr_auc']:.4f})", loc="left")
    ax[1].set_xlabel("Recall (fake)"); ax[1].set_ylabel("Precision (fake)"); ax[1].set_ylim(0.8, 1.005)
    for a in ax: a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(OUT / "fig_audit_calibration_pr.png", dpi=150); plt.close()
    print("calibration:", R["calibration"])

    # ---------- 6. generalisation: time split and topic shift ----------
    ok = dates.notna() & (X.str.strip() != "")
    cut = pd.Timestamp("2017-07-01")
    tr_m, te_m = ok & (dates < cut), ok & (dates >= cut)
    R["time_split"] = {"cutoff": str(cut.date()),
                       "train_class_counts": {"real": int((y[tr_m] == 0).sum()), "fake": int((y[tr_m] == 1).sum())},
                       "test_class_counts": {"real": int((y[te_m] == 0).sum()), "fake": int((y[te_m] == 1).sum())},
                       **fit_eval(X[tr_m], y[tr_m], X[te_m], y[te_m].values)}
    subj = df["subject"]
    tr_s = ok & subj.isin(["politicsNews", "politics", "left-news", "Government News"])
    te_s = ok & subj.isin(["worldnews", "News", "US_News", "Middle-east"])
    R["topic_shift"] = {"train_subjects": "real: politicsNews; fake: politics, left-news, Government News",
                        "test_subjects": "real: worldnews; fake: News, US_News, Middle-east",
                        **fit_eval(X[tr_s], y[tr_s], X[te_s], y[te_s].values)}
    print("time split:", fmt(R["time_split"]["accuracy"]), "| topic shift:", fmt(R["topic_shift"]["accuracy"]))

    R["runtime_seconds"] = round(time.time() - t0, 1)
    (OUT / "audit_metrics.json").write_text(json.dumps(R, indent=2, default=float))
    write_report(R)
    print("wrote audit/audit_report.md and audit/audit_metrics.json in", R["runtime_seconds"], "s")


def write_report(R):
    T, C = R["test_metrics"], R["test_metrics_without_near_duplicates"]
    row = lambda name, m: (f"| {name} | {m['n']:,} | {m['accuracy']:.4f} | {m['precision_fake']:.4f} | {m['recall_fake']:.4f} | "
                           f"{m['f1_fake']:.4f} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {m['false_positive_rate']:.4f} | {m['false_negative_rate']:.4f} |")
    head = ("| Evaluation | n | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |\n"
            "|---|---|---|---|---|---|---|---|---|---|")
    a = R["artefacts_percent_of_class"]
    cm = T["confusion"]
    lines = [
        "# ML audit — TF-IDF + Linear SVM (C = 5)", "",
        "Generated by `audit/ml_audit.py`. Every number below is computed by that script from the project's data and",
        "saved model; rerun it to reproduce. Positive class = fake.", "",
        "## 1. Split reproduction", "",
        f"Experiment 3's split was reproduced from the raw CSVs ({R['split']['raw_articles']:,} raw → {R['split']['after_cleaning']:,} after removing",
        f"empty and duplicate articles → {R['split']['train']:,} / {R['split']['val']:,} / {R['split']['test']:,}). "
        f"Matches the saved `test.csv`: **{R['split']['matches_saved_test_csv']}**.", "",
        "## 2. Duplicates and near-duplicates", "",
        f"- Identical cleaned texts: test also in train **{R['exact_duplicates']['test_in_train']}**, val in train "
        f"**{R['exact_duplicates']['val_in_train']}**, test in val **{R['exact_duplicates']['test_in_val']}** "
        f"(Experiment 3 de-duplicated on raw title + text, so texts that become identical after cleaning can survive: "
        f"{R['exact_duplicates']['duplicate_cleaned_texts_overall']} such duplicates overall; "
        f"{R['exact_duplicates']['empty_after_cleaning']} texts are empty after cleaning).",
        "- Test articles whose nearest training article has TF-IDF cosine similarity ≥ threshold: " +
        ", ".join(f"{k}: **{v}**" for k, v in R["near_duplicates_test_vs_train"].items()) + ".", "",
        "## 3. Held-out test metrics", "", head, row("Test set (as reported in Experiment 6)", T),
        row("Test set without near-duplicates (cos < 0.9)", C), "",
        f"Confusion matrix (test): TN {cm['tn']:,} · FP {cm['fp']:,} (real flagged fake) · FN {cm['fn']:,} (fake missed) · TP {cm['tp']:,}.", "",
        "## 4. Shortcuts and leakage in the dataset", "",
        "| Artefact | % of real | % of fake |", "|---|---|---|",
        *[f"| {k} | {v['real']} | {v['fake']} |" for k, v in a.items()], "",
        f"- `subject` values are disjoint between classes (overlap: {R['subjects_overlap'] or 'none'}) — real: {', '.join(R['subjects']['real'])}; "
        f"fake: {', '.join(R['subjects']['fake'])}. The model does not use this column, but any feature derived from it would be pure label leakage.",
        f"- Date ranges differ: real {R['date_range']['real'][0]} → {R['date_range']['real'][1]}, fake {R['date_range']['fake'][0]} → "
        f"{R['date_range']['fake'][1]} ({R['unparseable_dates']} unparseable dates).",
        f"- **Shortcut probe:** a TF-IDF + SVM trained on only the **first 12 words** of each article reaches accuracy "
        f"**{R['shortcut_first_12_words']['accuracy']:.4f}** (ROC-AUC {R['shortcut_first_12_words']['roc_auc']:.4f}). Datelines such as "
        "\"WASHINGTON (Reuters) -\" sit there, so the label is largely readable from the opening words.",
        f"- **Ablation:** deleting the single word \"said\" from test articles changes accuracy from {T['accuracy']:.4f} to "
        f"**{R['ablation_without_said']['accuracy']:.4f}** (FPR {R['ablation_without_said']['false_positive_rate']:.4f}).", "",
        "## 5. Calibration", "",
        "| Probability | Brier score | ECE (10 bins) |", "|---|---|---|",
        *[f"| {k} | {v['brier']:.4f} | {v['ece']:.4f} |" for k, v in R["calibration"].items()], "",
        "The SVM outputs a margin, not a probability. Neither row is a probability that an article is *true*: both only estimate",
        "P(label = fake) for articles drawn from this dataset. The UI therefore labels the bar as a writing-style score.", "",
        "![Reliability diagram and precision-recall curve](fig_audit_calibration_pr.png)", "",
        "## 6. Generalisation tests (retrained TF-IDF + Linear SVM, C = 5)", "", head,
        row(f"Time split: train before {R['time_split']['cutoff']}, test after", R["time_split"]),
        row("Topic shift: train on some subjects, test on others", R["topic_shift"]), "",
        f"Time split class counts — train: {R['time_split']['train_class_counts']}, test: {R['time_split']['test_class_counts']}. "
        f"Topic shift — {R['topic_shift']['train_subjects']} → {R['topic_shift']['test_subjects']}.", "",
    ]
    (OUT / "audit_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
