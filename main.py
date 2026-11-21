#!/usr/bin/env python3
"""Leak-free emotion recognition on SEED-IV (EEG + eye movements).

Runs ONE fold of ONE evaluation protocol, so every fold can be its own pod
in a Kubernetes Indexed Job on Nautilus (see k8s/loso-job.yaml).

Examples
--------
  # Smoke test on synthetic SEED-IV-shaped data (no dataset needed)
  python main.py --synthetic /tmp/seed_synth --protocol loso --fold 0

  # Check the layout of the real dataset on the PVC
  python main.py --data-dir /data/SEED_IV --inspect

  # One leave-one-subject-out fold on real data
  python main.py --data-dir /data/SEED_IV --protocol loso --fold 3 --model mlp

  # Summarize every fold written so far
  python main.py --aggregate --results-dir /data/results
"""

import argparse
import glob
import json
import os
import re
import socket
import sys
import time
from collections import Counter, defaultdict

import numpy as np
from scipy.io import loadmat, savemat
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Trial labels for each SEED-IV session, as published in the dataset ReadMe.
# 0 = neutral, 1 = sad, 2 = fear, 3 = happy
SESSION_LABELS = {
    1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
}
CLASS_NAMES = ["neutral", "sad", "fear", "happy"]
N_TRAIN_TRIALS = 16  # standard SEED-IV split: trials 1-16 train, 17-24 test
PROTOCOLS = ("pooled", "within", "loso", "loso_clip")


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def subject_id(path):
    """SEED-IV files are named <subject>_<date>.mat, e.g. 1_20160518.mat."""
    m = re.match(r"(\d+)_", os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot parse a subject id from {path}")
    return int(m.group(1))


def trial_arrays(mat, prefix):
    """Map trial number -> array for keys like de_LDS1..de_LDS24 or eye_1..eye_24."""
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    return {int(m.group(1)): np.asarray(v, dtype=float)
            for k, v in mat.items() if (m := pattern.match(k))}


def orient_eye(arr, n_windows):
    """Return eye features as (windows, features), whichever way they were stored."""
    arr = arr.reshape(arr.shape[0], -1) if arr.ndim != 2 else arr
    if arr.shape[0] == n_windows:
        return arr
    if arr.shape[1] == n_windows:
        return arr.T
    return arr if arr.shape[0] >= arr.shape[1] else arr.T


def load_seed_iv(data_dir, modality="eeg", feature="de_LDS", eye_prefix="eye_"):
    """Load SEED-IV feature files into window-level arrays with subject/session/trial ids."""
    X, y, subj, sess, trial = [], [], [], [], []
    mismatched = 0
    for session in (1, 2, 3):
        eeg_files = sorted(glob.glob(os.path.join(data_dir, "eeg_feature_smooth", str(session), "*.mat")))
        if not eeg_files:
            sys.exit(f"No EEG files in {data_dir}/eeg_feature_smooth/{session}/ (run with --inspect)")
        eye_files = {}
        if modality != "eeg":
            for f in glob.glob(os.path.join(data_dir, "eye_feature_smooth", str(session), "*.mat")):
                eye_files[subject_id(f)] = f
        for f in eeg_files:
            s = subject_id(f)
            eeg = trial_arrays(loadmat(f), feature)
            if not eeg:
                sys.exit(f"No '{feature}<n>' keys in {f} (run with --inspect)")
            eye = {}
            if modality != "eeg":
                if s not in eye_files:
                    sys.exit(f"No eye-movement file for subject {s}, session {session}")
                eye = trial_arrays(loadmat(eye_files[s]), eye_prefix)
            for t in sorted(eeg):
                a = eeg[t]                                   # (62 channels, W windows, 5 bands)
                n = a.shape[1]
                parts = [a.transpose(1, 0, 2).reshape(n, -1)]  # (W, 310)
                if modality != "eeg":
                    e = orient_eye(eye[t], n)                # (W, 31)
                    if e.shape[0] != n:
                        mismatched += 1
                        n = min(n, e.shape[0])
                    parts = [parts[0][:n], e[:n]]
                    if modality == "eye":
                        parts = parts[1:]
                X.append(np.hstack(parts))
                y += [SESSION_LABELS[session][t - 1]] * n
                subj += [s] * n
                sess += [session] * n
                trial += [t] * n
    if mismatched:
        print(f"warning: {mismatched} trials had different EEG/eye window counts; truncated to the shorter")
    return {"X": np.vstack(X), "y": np.array(y), "subj": np.array(subj),
            "sess": np.array(sess), "trial": np.array(trial)}


# ----------------------------------------------------------------------------
# Evaluation protocols
# ----------------------------------------------------------------------------
def n_folds(protocol, n_subjects):
    return {"pooled": 1, "within": 3 * n_subjects, "loso": n_subjects, "loso_clip": n_subjects}[protocol]


def make_split(d, protocol, fold, seed):
    """Boolean train/test masks over windows. Whole trials always stay on one side."""
    subjects = np.unique(d["subj"])
    if not 0 <= fold < n_folds(protocol, len(subjects)):
        sys.exit(f"--fold must be in [0, {n_folds(protocol, len(subjects)) - 1}] for {protocol}")
    if protocol == "pooled":
        # Same procedure as the original CMP_SC 4001/4720 notebook: stratified 80/20 split of whole
        # trials with every subject pooled, so each person appears in train AND test.
        tid = d["subj"] * 1000 + d["sess"] * 100 + d["trial"]
        trials, first = np.unique(tid, return_index=True)
        _, test_trials = train_test_split(trials, test_size=0.2, stratify=d["y"][first], random_state=seed)
        test = np.isin(tid, test_trials)
        return ~test, test, "all subjects pooled, 20% of trials held out"
    if protocol == "within":
        s, se = subjects[fold // 3], fold % 3 + 1
        mask = (d["subj"] == s) & (d["sess"] == se)
        return (mask & (d["trial"] <= N_TRAIN_TRIALS), mask & (d["trial"] > N_TRAIN_TRIALS),
                f"subject {s}, session {se}: trials 1-16 train, 17-24 test")
    s = subjects[fold]
    if protocol == "loso":
        return d["subj"] != s, d["subj"] == s, f"held-out subject {s}"
    # loso_clip: neither the person nor the film clip is seen in training
    return ((d["subj"] != s) & (d["trial"] <= N_TRAIN_TRIALS),
            (d["subj"] == s) & (d["trial"] > N_TRAIN_TRIALS),
            f"held-out subject {s} and held-out clips 17-24")


def subject_zscore(X, subj):
    """Z-score each subject with its own statistics (uses unlabeled test-subject data)."""
    X = X.copy()
    for s in np.unique(subj):
        m = subj == s
        X[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-8)
    return X


def build_model(name, seed):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    if name == "svm":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
    if name == "mlp":  # the original project's best configuration: PCA (99.9% variance) + MLP(100, 50)
        return make_pipeline(StandardScaler(), PCA(n_components=0.999, random_state=seed),
                             MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=seed))
    raise ValueError(name)


def evaluate(y_true, y_pred, trial_ids):
    labels = list(range(len(CLASS_NAMES)))
    votes, truth = defaultdict(list), {}
    for g, yt, yp in zip(trial_ids, y_true, y_pred):
        votes[g].append(yp)
        truth[g] = yt
    trial_hits = [Counter(v).most_common(1)[0][0] == truth[g] for g, v in votes.items()]
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "window_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "trial_accuracy": float(np.mean(trial_hits)),
        "per_class_recall": {c: float(r) for c, r in zip(CLASS_NAMES, recalls)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n_test_trials": len(votes),
    }


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------
def run_fold(args):
    t0 = time.time()
    d = load_seed_iv(args.data_dir, args.modality, args.feature, args.eye_prefix)
    train, test, desc = make_split(d, args.protocol, args.fold, args.seed)
    X = subject_zscore(d["X"], d["subj"]) if args.subject_norm else d["X"]
    model = build_model(args.model, args.seed)
    model.fit(X[train], d["y"][train])
    pred = model.predict(X[test])
    tid = d["subj"] * 1000 + d["sess"] * 100 + d["trial"]
    metrics = evaluate(d["y"][test], pred, tid[test])

    run_name = "_".join([args.modality] + ([args.feature] if args.modality != "eye" else []) + [args.model])
    run_name += "_snorm" if args.subject_norm else ""
    out_dir = os.path.join(args.results_dir, args.protocol, run_name)
    os.makedirs(out_dir, exist_ok=True)
    record = {
        "protocol": args.protocol, "fold": args.fold, "split": desc, "run": run_name,
        "n_train_windows": int(train.sum()), "n_test_windows": int(test.sum()),
        "n_features": int(X.shape[1]), "seed": args.seed,
        "runtime_seconds": round(time.time() - t0, 2), "host": socket.gethostname(),
        **metrics,
    }
    path = os.path.join(out_dir, f"fold_{args.fold:02d}.json")
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"[{args.protocol} fold {args.fold}] {desc} | {run_name} | "
          f"window acc {metrics['window_accuracy']:.3f}, macro-F1 {metrics['macro_f1']:.3f}, "
          f"trial acc {metrics['trial_accuracy']:.3f} | {record['runtime_seconds']}s -> {path}")


def aggregate(results_dir):
    groups = defaultdict(list)
    for path in glob.glob(os.path.join(results_dir, "*", "*", "fold_*.json")):
        with open(path) as fh:
            r = json.load(fh)
        groups[(r["protocol"], r["run"])].append(r)
    if not groups:
        sys.exit(f"No fold results under {results_dir}")
    rows = ["protocol,run,folds,window_acc_mean,window_acc_sd,macro_f1_mean,macro_f1_sd,trial_acc_mean,trial_acc_sd"]
    print(f"{'protocol':<10} {'run':<28} {'folds':>5}  {'window acc':>13}  {'macro-F1':>13}  {'trial acc':>13}")
    for (protocol, run), rs in sorted(groups.items()):
        stats = []
        for key in ("window_accuracy", "macro_f1", "trial_accuracy"):
            v = np.array([r[key] for r in rs])
            stats += [v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0]
        print(f"{protocol:<10} {run:<28} {len(rs):>5}  " +
              "  ".join(f"{stats[i]:.3f} ± {stats[i + 1]:.3f}" for i in (0, 2, 4)))
        rows.append(",".join([protocol, run, str(len(rs))] + [f"{x:.4f}" for x in stats]))
    with open(os.path.join(results_dir, "summary.csv"), "w") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"wrote {os.path.join(results_dir, 'summary.csv')}")


def inspect(data_dir):
    for folder in ("eeg_feature_smooth", "eye_feature_smooth"):
        for session in (1, 2, 3):
            files = sorted(glob.glob(os.path.join(data_dir, folder, str(session), "*.mat")))
            print(f"{folder}/{session}: {len(files)} files, subjects {sorted(subject_id(f) for f in files)}")
            if files and session == 1:
                mat = loadmat(files[0])
                keys = sorted(k for k in mat if not k.startswith("__"))
                print(f"  {os.path.basename(files[0])}: {len(keys)} keys, e.g. {keys[:6]}")
                for k in keys[:2]:
                    print(f"  {k}: shape {np.asarray(mat[k]).shape}")


def make_synthetic(out_dir, n_subjects=15, seed=0):
    """Write small SEED-IV-shaped feature files with a weak class signal plus subject offsets."""
    if glob.glob(os.path.join(out_dir, "eeg_feature_smooth", "1", "*.mat")):
        return
    rng = np.random.default_rng(seed)
    # Class signal = a weak pattern shared by everyone + a stronger person-specific one,
    # so (as with real EEG) models do better within a subject than on new subjects.
    eeg_shared, eye_shared = rng.normal(size=(4, 62, 5)), rng.normal(size=(4, 31))
    for s in range(1, n_subjects + 1):
        eeg_own, eye_own = rng.normal(size=(4, 62, 5)), rng.normal(size=(4, 31))
        eeg_offset, eye_offset = rng.normal(scale=1.5, size=(62, 5)), rng.normal(scale=1.5, size=31)
        for session in (1, 2, 3):
            eeg_mat, eye_mat = {}, {}
            for t in range(1, 25):
                c, w = SESSION_LABELS[session][t - 1], int(rng.integers(8, 15))
                trial_noise = rng.normal(scale=0.5, size=(62, 5))  # clip-to-clip variation
                eeg = (0.06 * eeg_shared[c] + 0.12 * eeg_own[c] + eeg_offset + trial_noise
                       + rng.normal(size=(w, 62, 5)))
                eye = 0.25 * eye_shared[c] + 0.25 * eye_own[c] + eye_offset + rng.normal(size=(w, 31))
                eeg_mat[f"de_LDS{t}"] = eeg.transpose(1, 0, 2)   # (62, W, 5) like SEED-IV
                eye_mat[f"eye_{t}"] = eye.T                        # (31, W)
            for folder, mat in (("eeg_feature_smooth", eeg_mat), ("eye_feature_smooth", eye_mat)):
                path = os.path.join(out_dir, folder, str(session))
                os.makedirs(path, exist_ok=True)
                savemat(os.path.join(path, f"{s}_2016010{session}.mat"), mat)
    print(f"wrote synthetic SEED-IV-shaped data to {out_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="/data/SEED_IV", help="folder holding eeg_feature_smooth/ and eye_feature_smooth/")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--protocol", choices=PROTOCOLS, default="loso")
    p.add_argument("--fold", type=int, default=0, help="fold index (one pod per fold)")
    p.add_argument("--modality", choices=("eeg", "eye", "fusion"), default="eeg")
    p.add_argument("--model", choices=("logreg", "svm", "mlp"), default="logreg")
    p.add_argument("--feature", default="de_LDS", help="EEG key prefix: de_LDS, de_movingAve, psd_LDS, psd_movingAve")
    p.add_argument("--eye-prefix", default="eye_", help="eye-movement key prefix in the .mat files")
    p.add_argument("--subject-norm", action="store_true", help="z-score each subject with its own statistics")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inspect", action="store_true", help="print the dataset layout and exit")
    p.add_argument("--aggregate", action="store_true", help="summarize fold results and exit")
    p.add_argument("--synthetic", metavar="DIR", help="generate synthetic data in DIR and use it")
    args = p.parse_args()

    if args.aggregate:
        return aggregate(args.results_dir)
    if args.synthetic:
        make_synthetic(args.synthetic)
        args.data_dir = args.synthetic
    if args.inspect:
        return inspect(args.data_dir)
    run_fold(args)


if __name__ == "__main__":
    main()
