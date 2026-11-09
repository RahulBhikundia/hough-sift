# Hough-SIFT: Hough pose clustering for learned features with semantic-gated voting

Computer Vision course project, IIT Jodhpur, November 2026.
Rahul Kumar Bhikundia (M26CSA006), Suggu Chandu (M26CSA017), Navneet Singh (M26CSA004).

Report: `report/main.pdf` (LaTeX source `report/main.tex`).

## Layout

| Path | Contents |
|---|---|
| `hough_sift/features.py` | SIFT (RootSIFT) and DISK features, ratio + mutual-NN matching |
| `hough_sift/hough.py` | Single-match (Lowe) and pairwise similarity voting, 4-D/5-D Hough accumulator |
| `hough_sift/verify.py` | Deterministic cluster refinement (affine to homography), RANSAC / MAGSAC++ baselines |
| `hough_sift/semantic.py` | Mask R-CNN instance maps and semantic gating rules R1-R3 (soft/strict) |
| `hough_sift/hpatches.py` | HPatches loader and corner-error metric |
| `scripts/` | Experiments and figures (see below) |
| `results/` | Final CSVs and figures; `results/pilot/` holds early pilot runs |

## Setup

Python 3.12, CUDA GPU optional (tested on a GTX 1650, 4 GB).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Data (not included, about 2 GB):

```bash
# HPatches sequences -> data/hpatches_raw/hpatches-sequences-release/
curl -L -o hpatches.zip https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.zip
mkdir -p data/hpatches_raw && python -m zipfile -e hpatches.zip data/hpatches_raw

# COCO val2017 images + instance annotations -> data/coco/
curl -LO http://images.cocodataset.org/zips/val2017.zip
curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip
mkdir -p data/coco && python -m zipfile -e val2017.zip data/coco
python -c "import zipfile; zipfile.ZipFile('annotations_trainval2017.zip').extract('annotations/instances_val2017.json', 'data/coco')"
```

DISK and Mask R-CNN weights download automatically on first use.

## Reproducing the results

Run one at a time (each uses up to about 2.3 GB RAM).

```bash
python scripts/sanity_synthetic.py                     # quick check on a synthetic warp
python scripts/eval_hpatches.py --outlier-frac 0       # HPatches, repeat for 0.5 0.8 0.9 0.95
python scripts/sweep_k.py --limit 20                   # neighbourhood-size sensitivity
python scripts/eval_multiobject.py --n 100 --offset 150   # validation split (design decisions)
python scripts/eval_multiobject.py --n 150 --outlier-frac 0   # test split, repeat for 0.5 0.8
python scripts/make_figures.py                         # figures into results/figures/
python scripts/visualize_multiobj.py --n 15            # qualitative examples
```

Build the report: `cd report && latexmk -pdf main.tex`.
