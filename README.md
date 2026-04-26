# Aphasia Detection and Severity Prediction

An end-to-end ML pipeline for automatic detection and severity classification of aphasia in English-speaking individuals, built on acoustic and linguistic features extracted from clinical speech data.

Inspired by [Ying Qin et al. (arXiv:1904.00361)](https://arxiv.org/abs/1904.00361) — originally designed for Cantonese, adapted here for English using the AphasiaBank corpus.

---

## What it does

Takes utterance-level features (MFCCs, pitch, jitter, shimmer, formants, speech rate, etc.) extracted from patient speech and classifies aphasia severity as **Mild** or **Severe** based on Aphasia Quotient (AQ) scores.

Three architectures were trained and compared:

| Model | Accuracy | AUC Score |
|-------|----------|-----------|
| CNN | 0.9247 | 0.8045 |
| GRU | 0.9328 | 0.8470 |
| Hybrid CNN+GRU | 0.9301 | 0.8744 |

GRU achieved the highest accuracy. Hybrid CNN+GRU achieved the highest AUC score.

---

## Repo structure

```
├── data/sample/          # Sample CSV (1 speaker, for reference)
├── src/
│   └── cha_to_csv.py     # Pipeline to extract features from .cha transcripts
├── m3_aphasia.ipynb      # Full training notebook (CNN, GRU, Hybrid)
└── README.md
```

---

## Pipeline overview

1. `.cha` transcript files parsed → utterance-level acoustic and linguistic features extracted using `librosa` and `praat-parselmouth` (note: features are derived from transcripts given along .wav audio files in dataset)
2. Features exported to CSV 
3. Patient-only rows filtered, AQ threshold (90) applied for binary labels
4. Models trained with stratified splits, evaluated on accuracy + AUC

---

## Dataset

Data sourced from the [AphasiaBank English corpus](https://sla.talkbank.org/TBB/aphasia/) — **not included in this repo**.

Access requires institutional approval from TalkBank. Raw `.cha` and `.wav` files are restricted under their data use agreement.

The `data/sample/` folder contains one anonymized CSV for reference only.

---

## Run it

```bash
# 1. Extract features from .cha files
python src/cha_to_csv.py --input_dir /path/to/cha_files --output_dir /path/to/output

# 2. Open and run the notebook
jupyter notebook m3_aphasia.ipynb
```

```bash
pip install -r requirements.txt
```

---

## Acknowledgements

This work was carried out during a summer research internship under the supervision of **Dr. Shikha Baghel**, Dept. of Electrical Engineering, IIT Jammu (earlier at IIT Tirupati).

---

## ⚠️ Data Notice

This repository contains **no clinical data**. The AphasiaBank corpus is a restricted dataset — use of its data requires formal approval from [TalkBank](https://talkbank.org/). Do not attempt to reproduce or redistribute the original recordings or transcripts.

---

## License

Licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.
Code is open for use and modification. Dataset and clinical data are **not covered** under this license.
