# EEG + Eye-Movement Emotion Recognition on Nautilus

Four-class emotion recognition (neutral, sad, fear, happy) on the SEED-IV dataset, built to answer one question: **do the models still work on people and film clips they were never trained on?** Every cross-validation fold runs as its own pod in a Kubernetes Job on Nautilus (NRP).

DATA SCI 8420 final project. Bach Nguyen (bxnwx2).

## Background and attribution

This project extends the CMP_SC 4001/4720 group project *Brainwaves and Emotions: Mapping the Mind with Machine Learning* (Joshua Green, Bach Nguyen, Bond Leung, University of Missouri, 2025) and its FABRIC pipeline repository, [JoshGreen000/4001_CI_Bio_Eng_Final_Project](https://github.com/JoshGreen000/4001_CI_Bio_Eng_Final_Project). That project reported 69% accuracy (PCA + MLP on SEED-IV differential-entropy features) using a split in which every subject appeared in both the training and the test data.

The code in this repository is new. It re-implements the earlier best model (`--protocol pooled --model mlp`) only as a reference point. Anything copied from the original repository is listed here with its source commit and my teammates' permission:

- *(nothing copied yet)*

## What's new compared with the original project

- **Leak-free evaluation.** Four protocols, from the original pooled split to leave-one-subject-out with held-out film clips (table below).
- **Real multimodal fusion.** SEED-IV's eye-movement features are recorded at the same time as the EEG, from the same people, so the two can be fused window by window (`--modality fusion`). The earlier EEG and fMRI data came from different people and could not be.
- **Cloud-native runs.** One pod per fold (Kubernetes Indexed Job) instead of bash loops limited to three parallel jobs on one VM.

## Data (not in this repo)

SEED-IV comes from the BCMI Lab at Shanghai Jiao Tong University; access is granted on request under a license agreement. Keep it on the PVC only:

```text
/data/SEED_IV/
├─ eeg_feature_smooth/{1,2,3}/<subject>_<date>.mat   # de_LDS1..de_LDS24, each (62 channels, windows, 5 bands)
└─ eye_feature_smooth/{1,2,3}/<subject>_<date>.mat   # eye-movement features (31 per window)
```

Run `python main.py --data-dir /data/SEED_IV --inspect` first to confirm the key names and shapes. Eye keys are assumed to be `eye_1` … `eye_24`; change that with `--eye-prefix` if needed.

**Never commit SEED-IV files, or anything derived from them, to GitHub or GitLab.** `.gitignore` blocks `data/` and `*.mat`.

## Quick start

```bash
pip install -r requirements.txt
python main.py --synthetic /tmp/seed_synth --protocol loso --fold 0   # synthetic data, no download needed
python main.py --data-dir /data/SEED_IV --protocol loso --fold 0 --model mlp
python main.py --aggregate --results-dir results
```

## Evaluation protocols

| `--protocol` | Train on | Test on | Folds |
|---|---|---|---|
| `pooled` | 80% of trials, all subjects mixed | remaining 20% (same people) | 1 (the original setup) |
| `within` | one subject's trials 1–16 in one session | that subject's trials 17–24 | 45 (standard SEED-IV protocol) |
| `loso` | 14 subjects | the held-out subject | 15 |
| `loso_clip` | 14 subjects, clips 1–16 | held-out subject, clips 17–24 | 15 |

Whole trials always stay on one side of a split. Each fold writes a JSON file with window accuracy, macro-F1, per-class recall, trial-level (majority-vote) accuracy, the confusion matrix, runtime and pod name.

- `--subject-norm` z-scores each subject with its own statistics. That uses the held-out subject's unlabeled features, so report it as a separate setting.
- If you tune hyperparameters, tune inside the training data with `GroupKFold` over trials or subjects, never over individual windows.

## Running on Nautilus

1. Clone this repo into the PVC (Practice 5.1) so it sits at `/data/project1/eeg-emotion-nautilus`.
2. Put SEED-IV at `/data/SEED_IV`.
3. `kubectl -n gp-engine-mizzou-dsa-cloud create -f k8s/loso-job.yaml` (15 pods, one per held-out subject).
4. When the pods finish: `python main.py --aggregate --results-dir /data/results`.

## Roadmap

- [ ] LSTM over consecutive 4-second windows within a trial (GPU Job)
- [ ] EEG + eye fusion: concatenation (done), CCA shared embedding, deep CCA (stretch)
- [ ] Docker image built by GitLab CI and stored in the NRP registry
- [ ] Cross-dataset check on GAMEEMO (stretch)
