"""Build and execute the four walkthrough notebooks from source definitions.

Keeping the notebooks generated means they cannot drift out of sync with the
code, and executing them means the committed .ipynb files show real outputs on
GitHub instead of empty cells.

    python scripts/build_notebooks.py

Design rules for these notebooks, in priority order:

1. **Readable without running.** Every cell is executed here and its output is
   committed, so a reviewer sees real numbers and real figures on GitHub.
2. **Runnable if you want to.** They operate on `data/sample/` (46 committed
   images) and the committed CSVs, so they work on a clean checkout with no
   dataset download and no GPU.
3. **No API calls, ever.** Question-generation output is read from committed
   results. Nothing here spends anyone's quota or needs a key.

Notebook 02 has an opt-in `RUN_LIVE_DETECTION` flag that runs OWL-ViT over the
sample images. It defaults to False because the model is a ~600 MB download.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[1]
NB_DIR = REPO_ROOT / "notebooks"

BOOTSTRAP = """\
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
pd.set_option("display.width", 140)
pd.set_option("display.max_colwidth", 90)

SAMPLE = REPO_ROOT / "data" / "sample"
RESULTS = REPO_ROOT / "results"
print("Repository root:", REPO_ROOT.name)
"""

# Shared plotting helper. Kept in one place so all four notebooks draw boxes
# the same way.
DRAW_HELPER = '''\
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

# Photographs re-encode as PNG in notebook output, which is what drives .ipynb
# file size. 72 dpi keeps the committed notebooks light enough to render quickly
# on GitHub while staying perfectly legible.
plt.rcParams["figure.dpi"] = 72


def load_sample_coco(amenity):
    """Ground-truth boxes for the committed sample, keyed by file name."""
    coco = json.loads((SAMPLE / amenity / "_annotations.coco.json").read_text())
    by_id = {i["id"]: i["file_name"] for i in coco["images"]}
    boxes = {}
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        boxes.setdefault(by_id[ann["image_id"]], []).append([x, y, x + w, y + h])
    return coco, boxes


def show_detections(amenity, file_names, detections, gt_boxes, conf, ncols=3):
    """Green = ground truth, orange = detection above `conf`."""
    nrows = (len(file_names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, name in zip(axes, file_names):
        ax.imshow(Image.open(SAMPLE / amenity / name).convert("RGB"))
        for x1, y1, x2, y2 in gt_boxes.get(name, []):
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, edgecolor="#2E9B57", linewidth=2.5))
        hits = detections[(detections["image"] == name) & (detections["score"] > conf)]
        for _, d in hits.iterrows():
            ax.add_patch(Rectangle((d.x1, d.y1), d.x2 - d.x1, d.y2 - d.y1,
                                   fill=False, edgecolor="#E07B39", linewidth=2.0,
                                   linestyle="--"))
            ax.text(d.x1, max(d.y1 - 6, 10), f"{d.score:.2f}", color="#E07B39",
                    fontsize=9, fontweight="bold")
        n_gt = len(gt_boxes.get(name, []))
        ax.set_title(f"{n_gt} ground truth · {len(hits)} detected", fontsize=9)
        ax.axis("off")

    for ax in axes[len(file_names):]:
        ax.axis("off")
    fig.suptitle("green = ground truth    orange dashed = OWL-ViT detection",
                 fontsize=10, y=1.005)
    plt.tight_layout()
    plt.show()
'''


NOTEBOOKS = {
    # ------------------------------------------------------------------ 01
    "01_dataset_and_configuration.ipynb": [
        ("md", """# 01 · The problem, the data, and the configuration

**The question this project asks:** given a photograph from an accommodation
listing, can a system work out which questions a traveller would actually want
answered before booking?

The pipeline has two stages. A zero-shot object detector finds amenities in the
photo: a bathtub, a kettle, a TV. A language model then turns each detection
into a question a guest might ask. The interesting part is not either stage
alone; it is what has to be true for the handoff between them to work.

This notebook covers the setup: what is configured, what the data looks like, and
what ships in this repository.

---

### A note on scope

This repository is a **portfolio walkthrough**, not the production system. The
full-scale experiments ran on the industry collaborator's internal
infrastructure; that code, its credentials, and its configuration are not
published here. What you see is the method, reimplemented against public APIs and
run over a small committed sample, alongside the **real results** from the
original runs.

Everything numeric in these notebooks comes from the committed CSVs. They are the actual
measurements, not re-simulated ones."""),
        ("code", BOOTSTRAP),

        ("md", """## One code path, five amenities

The single biggest engineering decision in this project: every amenity runs
through the *same* code. The only things that differ are the text prompts and
the confidence sweep, and those live in `configs/amenities.yaml`.

The earlier version of this project had one ~1,200-line script per amenity, and
they were 86–99% identical to each other. Adding a sixth amenity now means adding
a YAML block."""),
        ("code", """from src.config import load_config

configs = load_config()

pd.DataFrame([
    {
        "amenity": key,
        "prompt sets": ", ".join(cfg.prompt_sets),
        "widest set": max(len(p) for p in cfg.prompt_sets.values()),
        "confidence sweep": f"{min(cfg.conf_sweep)} – {max(cfg.conf_sweep)}",
        "IoU threshold": cfg.eval_iou_threshold,
    }
    for key, cfg in configs.items()
])"""),

        ("md", """## Prompt design

The detection-side research question is whether a richer text prompt helps an
open-vocabulary detector. Three families were tested:

| Family | Idea |
|:---|:---|
| `baseline` | the bare class name, `"kettle"` |
| `long` | one descriptive sentence |
| `variants` | several short synonyms and sub-types at once |

Intuition says `long` should win: more description, more signal for the text
encoder. Notebook 02 shows that intuition is wrong."""),
        ("code", """for key in ["kettle", "bathtub"]:
    cfg = configs[key]
    print(f"=== {cfg.display_name} ===")
    for name, prompts in cfg.prompt_sets.items():
        print(f"  {name:10s} {prompts}")
    print()"""),

        ("md", """## Where the data comes from

The images are **real accommodation listing photographs provided by trivago
N.V.**, reproduced with permission. Not a general-purpose object-detection
benchmark repurposed for a hotel task; the actual domain the system targets.
A small number of images in the TV set are public stock photography.

Building the dataset was most of the work. The inventory carries coarse room
tags (bathroom, bedroom, kitchen) but no object-level labels, so images
containing a bathtub could not simply be queried. Candidates were pulled by room
tag and then reviewed by hand, one at a time; ambiguous, duplicated and corrupted
images were discarded. All metadata except the image itself was then stripped, so
no textual signal could leak into a pipeline meant to work from pixels alone.
Bounding boxes were drawn manually in a private Roboflow workspace.

**No annotated data trains anything.** OWL-ViT runs zero-shot; the annotations
exist purely to evaluate it.

The full splits are ~1,500 images and are not committed. A deterministic
46-image sample is, with attribution in
[`data/sample/ATTRIBUTION.md`](../data/sample/ATTRIBUTION.md)."""),
        ("code", DRAW_HELPER + """
rows = []
for key in configs:
    coco, boxes = load_sample_coco(key)
    n_images = len(coco["images"])
    n_boxes = sum(len(v) for v in boxes.values())
    rows.append({
        "amenity": key,
        "sample images": n_images,
        "ground-truth boxes": n_boxes,
        "images with no box": n_images - len(boxes),
    })

pd.DataFrame(rows)"""),

        ("md", """Images carrying no ground-truth box are deliberate. They are the **negative
set**: rooms where the amenity is genuinely absent. Precision and recall on
positive images say nothing about how often a detector hallucinates a kettle in a
bathroom, so those images are scored separately in notebook 02."""),

        ("md", """## What the data actually looks like

Worth looking at before trusting any metric computed from it."""),
        ("code", """coco, gt = load_sample_coco("bathtub")
names = [i["file_name"] for i in coco["images"] if i["file_name"] in gt][:6]

empty = pd.DataFrame(columns=["image", "score", "x1", "y1", "x2", "y2"])
show_detections("bathtub", names, empty, gt, conf=1.0)"""),

        ("md", """These are the honest limits of the dataset, and they bound everything that
follows:

- **Annotation tightness varies.** Some boxes hug the tub, others take in the
  surrounding tiling. That caps achievable IoU regardless of how good the
  detector is, and is part of why mean IoU sits well below F1.
- **The validation splits are small**, 18 images for bathtub and 52 for mirror.
  Too small for confident model selection, which is why threshold choices are
  reported as a sweep rather than a single tuned number.
- **A narrow slice of the domain.** Well-curated listings in three European
  cities, not a random sample of the inventory. Performance on lower-quality
  photography, or in other markets, is untested.

[`docs/dataset_inventory.md`](../docs/dataset_inventory.md) records the full
split sizes, provenance, and the integrity issues found during the audit."""),
    ],

    # ------------------------------------------------------------------ 02
    "02_detection_pipeline.ipynb": [
        ("md", """# 02 · Zero-shot amenity detection with OWL-ViT

[OWL-ViT](https://huggingface.co/google/owlvit-base-patch16) detects objects
described by text, with no fine-tuning and no training data. You give it an
image and a list of phrases; it returns boxes and scores.

That property is what makes the whole project tractable: adding a new amenity
costs a line of YAML, not an annotation campaign.

```
image ──► OWL-ViT(text prompts) ──► boxes + scores
                                        │
                     confidence sweep ──┤
                                        ▼
              greedy IoU matching vs COCO ground truth
                                        │
                                        ▼
                    precision · recall · F1 · mean IoU
```

Inference runs **once** at a permissive threshold (0.05). The sweep then filters
those saved detections, so exploring 10 thresholds costs no additional GPU time.
Structuring it that way is why the full sweep was affordable at all.

Scoring applies class-agnostic NMS, then matches predictions one-to-one against
ground truth in descending confidence order, so several boxes on the same object
count once rather than several times. `tests/test_metrics.py` asserts the
invariant that makes this checkable: `tp + fn` always equals the ground-truth box
count, at every threshold and for every prompt set."""),
        ("code", BOOTSTRAP),
        ("code", DRAW_HELPER),
        ("code", """from src.config import load_config

configs = load_config()
print("Amenities configured:", ", ".join(configs))"""),

        ("md", """## Optionally: run the detector live

Everything below this cell reads the **committed** detection CSVs from the
original full-scale runs. If you want to watch OWL-ViT actually work, set the
flag and re-run. It will detect over the 10 committed bathtub sample images.

It is off by default because the model weights are a ~600 MB download, and
because the committed results are what the reported numbers are based on."""),
        ("code", """RUN_LIVE_DETECTION = False   # set True to run OWL-ViT on data/sample/bathtub

if RUN_LIVE_DETECTION:
    from src.detection.detector import OwlViTDetector

    detector = OwlViTDetector()                       # picks cuda / mps / cpu
    paths = sorted((SAMPLE / "bathtub").glob("*.jpg"))
    live = detector.detect_folder(
        paths,
        prompts=configs["bathtub"].prompt_sets["baseline"],
        threshold=0.05,
    )
    print(f"{len(live)} detections over {len(paths)} sample images")
    display(live.head())
else:
    print("Live detection skipped. The cells below use the committed detection CSVs.")
    print("Set RUN_LIVE_DETECTION = True to run OWL-ViT over data/sample/bathtub.")"""),

        ("md", """## Real detections on real images

These boxes are from the committed results of the original run, not
regenerated. Green is ground truth, orange dashed is what OWL-ViT found at the
threshold that maximised F1 for this amenity."""),
        ("code", """detections = pd.read_csv(RESULTS / "detection_outputs" / "bathtub" / "dets_bathtub_baseline.csv")
coco, gt = load_sample_coco("bathtub")

with_gt = [i["file_name"] for i in coco["images"] if i["file_name"] in gt][:6]
show_detections("bathtub", with_gt, detections, gt, conf=0.05)"""),

        ("md", """Two things are visible here that a metric table cannot show you: the detector
often finds the object but draws a looser box than the annotator did, and it
sometimes fires twice on one tub. Both depress IoU without meaning the detection
was useless."""),

        ("md", """## The confidence sweep

For one amenity and one prompt set, every threshold from 0.05 to 0.50."""),
        ("code", """from src.detection.metrics import best_per_prompt, load_summary

summary = load_summary(RESULTS / "detection_metrics" / "bathtub" / "summary_bathtub.csv")
summary[summary["prompt"] == "baseline"].sort_values("conf")[
    ["prompt", "conf", "precision", "recall", "f1", "mean_iou"]
]"""),

        ("md", """## Finding 1: richer prompts mostly did not help

Best row per prompt set, all five amenities."""),
        ("code", """rows = []
for key, cfg in configs.items():
    s = load_summary(RESULTS / "detection_metrics" / key / f"summary_{key}.csv")
    for _, best in best_per_prompt(s).iterrows():
        rows.append({
            "amenity": cfg.display_name,
            "prompt set": best["prompt"],
            "conf": best["conf"],
            "f1": round(best["f1"], 3),
        })

comparison = pd.DataFrame(rows).pivot(index="amenity", columns="prompt set", values="f1")
comparison"""),
        ("code", """winners = comparison.idxmax(axis=1).value_counts()
print("Winning prompt set, by amenity:")
print(winners.to_string())
print()
if "long" in comparison.columns:
    gap = (comparison["baseline"] - comparison["long"]).dropna()
    print(f"baseline beats long on {(gap > 0).sum()} of {len(gap)} amenities "
          f"(mean F1 gap {gap.mean():+.3f})")"""),

        ("md", """The bare class name wins for bathtub, mirror and TV, and the long descriptive
prompt is the *worst* option in every case. Hairdryer improves with multiple
synonyms, which makes sense, since "hairdryer", "hair dryer" and "blow dryer" are
genuinely different strings for the same object. Kettle is a tie inside the
noise.

The likely reason the long prompts hurt: OWL-ViT's text encoder embeds the whole
phrase, so "white ceramic bathtub in a hotel bathroom" is not a sharper bathtub
query. It is a different, more diffuse one, pulled toward whatever "hotel" and
"ceramic" mean in embedding space.

**Prompt engineering is not a free lunch for open-vocabulary detection.**"""),

        ("md", """## Finding 2: the best threshold is far lower than any sensible default"""),
        ("code", """best = []
for key, cfg in configs.items():
    s = load_summary(RESULTS / "detection_metrics" / key / f"summary_{key}.csv")
    top = best_per_prompt(s).sort_values("f1", ascending=False).iloc[0]
    best.append({
        "amenity": cfg.display_name,
        "best prompt": top["prompt"],
        "conf": top["conf"],
        "precision": round(top["precision"], 3),
        "recall": round(top["recall"], 3),
        "f1": round(top["f1"], 3),
        "mean IoU": round(top["mean_iou"], 3),
    })

headline = pd.DataFrame(best).sort_values("f1", ascending=False)
headline"""),
        ("code", """n_low = (headline["conf"] <= 0.05).sum()
print(f"{n_low} of {len(headline)} amenities peak at conf = 0.05, "
      f"{(headline['conf'] <= 0.15).sum()} at 0.15 or below")
print(f"Mean best F1 {headline['f1'].mean():.3f}  "
      f"(range {headline['f1'].min():.3f}–{headline['f1'].max():.3f})")

# What a conventional threshold would have cost, computed rather than asserted.
import statistics

for target in (0.25, 0.35):
    at_conf = []
    for key, cfg in configs.items():
        s = load_summary(RESULTS / "detection_metrics" / key / f"summary_{key}.csv")
        best_prompt = s.sort_values("f1", ascending=False).iloc[0]["prompt"]
        rows = s[(s["prompt"] == best_prompt) & (s["conf"].between(target - 0.01, target + 0.01))]
        if not rows.empty:
            at_conf.append(rows.iloc[0]["f1"])
    if at_conf:
        print(f"Mean F1 at conf {target:.2f}: {statistics.mean(at_conf):.3f}")

print(f"Mean F1 at the swept optimum: {headline['f1'].mean():.3f}")"""),

        ("md", """Every amenity peaks at 0.15 or below, most at 0.05. These are small, frequently
occluded objects in cluttered rooms, and OWL-ViT is systematically
*underconfident* about them. Mean F1 falls from 0.751 at the swept optimum to
0.540 at 0.25 and 0.333 at 0.35. Anyone who deployed this at a conventional
threshold would have concluded the approach does not work.

That is the practical lesson worth carrying out of this project: for
open-vocabulary detection on a new domain, the threshold is not a
hyperparameter to leave at its default. It is the first thing to measure."""),

        ("md", """![F1 across prompt sets and thresholds](../results/figures/bathtub/prompt_comparison_full.png)"""),

        ("md", """## The cost side: false positives on negative images

F1 on positive images says nothing about how often the detector invents an
amenity that is not there. Those images are scored separately."""),
        ("code", """negatives = pd.read_csv(
    RESULTS / "detection_metrics" / "bathtub" / "negatives_bathtub.csv",
    encoding="utf-8-sig",
)
negatives"""),

        ("md", """And there is the trade-off, stated plainly: **the low threshold that maximises
F1 is also the threshold with the weakest specificity.** A production system
would not pick one number for all five amenities. It would tune each one against
the cost of showing a traveller a question about a bathtub that does not exist,
which is a product decision, not a modelling one."""),
    ],

    # ------------------------------------------------------------------ 03
    "03_question_generation.ipynb": [
        ("md", """# 03 · From detection to question

A detection is a box, a label, and a score. A traveller wants a question. This
notebook is about the gap between those two things, and it contains the most
useful result in the project, which is a **negative** one.

All generated text below is read from committed results. Nothing here calls an
API or needs a key."""),
        ("code", BOOTSTRAP),
        ("code", DRAW_HELPER),

        ("md", """## The first approach: describe the detection in words

The obvious design. Turn the detection row into a short textual summary, wrap it
in a system instruction, send it to a language model.

One detail worth noting: the raw confidence score is **bucketed** rather than
passed through verbatim. Handing a model `0.7214` invited it to invent matching
precision in the question text ("is this likely-to-be-a-bathtub..."), so it
receives a band instead."""),
        ("code", """from src.question_generation.prompts import build_prompt, detection_summary

detections = pd.read_csv(RESULTS / "detection_outputs" / "bathtub" / "dets_bathtub_baseline.csv")
row = detections.sort_values("score", ascending=False).iloc[0]

print(detection_summary(row))
print()
print("-" * 78)
print(build_prompt(detection_summary(row)))"""),

        ("md", """## Finding 3: text-only conditioning collapses into templates

Here is the problem, and it is visible without any metric. These are real
generated questions from four backends over the same 1,003 detections."""),
        ("code", """generated = pd.read_csv(
    RESULTS / "question_generation_outputs" / "bathtub"
    / "merged_detections_bathtub_with_questions_all_models.csv"
)
question_cols = [c for c in generated.columns if c.startswith("generated_question")]

print(f"{len(generated)} detections, {len(question_cols)} backends\\n")
generated[["image"] + question_cols].head(5)"""),

        ("md", """Different images. Different confidence scores. Nearly identical questions.

Counting how often each backend repeats itself makes the scale of it obvious:"""),
        ("code", """for col in question_cols:
    values = generated[col].dropna().astype(str)
    top, count = values.value_counts().index[0], values.value_counts().iloc[0]
    backend = col.replace("generated_question_", "")
    print(f"{backend:14s} {count:>4} / {len(values)} rows ({count/len(values):>5.1%})   \\"{top[:58]}\\"")"""),

        ("md", """**Why this happens.** Look again at the prompt above. It carries
`label + confidence band + prompt type`. Across 1,003 bathtub detections that
input is nearly identical every time. The label is always "bathtub", the
confidence band takes three values, the prompt type one. The model is being asked
the same question a thousand times, so it gives the same answer.

The detection is *information-poor by construction*. Everything that makes one
bathtub different from another (its size, its condition, whether there is a
shower over it) was thrown away the moment the detection became a text summary."""),

        ("md", """### The metric that hides it

This is the part worth dwelling on. Standard question-quality metrics score this
output as a complete success."""),
        ("code", """from src.evaluation import question_metrics as qm

rows = []
for col in question_cols:
    stats = qm.summarize(generated[col].dropna().astype(str).tolist())
    stats["backend"] = col.replace("generated_question_", "")
    rows.append(stats)

quality = pd.DataFrame(rows).set_index("backend")
quality[["n", "fallback_rate", "well_formed_rate", "distinct_2", "duplicate_rate"]]"""),

        ("md", """Every backend scores **0% fallback** and **100% well-formedness**. By those two
measures the system works perfectly.

Well-formedness asks "is this shaped like a question?" It cannot see that it is
the *same* question 592 times. Only `distinct_2` and `duplicate_rate` catch it.

The generalisable lesson: **any evaluation of generated text needs a diversity
term, or it will report success on a system that has collapsed.** A dashboard
tracking only fluency and fallback rate would have shown green throughout."""),

        ("md", """## The fix: condition on the image, not a description of it

If the problem is that the text summary discards what makes each detection
distinct, the fix is to stop discarding it and pass the model the actual pixels.

The detected region is cropped with 35% padding on each side. The padding
matters: a tight crop of a kettle is just a kettle, while the surrounding context
is what lets a model ask about counter space or reachability."""),
        ("code", """from src.question_generation.prompts import pad_box

coco, gt = load_sample_coco("bathtub")
sample_names = {i["file_name"] for i in coco["images"]}

vision = pd.read_csv(
    RESULTS / "question_generation_outputs" / "bathtub"
    / "dets_bathtub_baseline_gemini3_perspectives.csv"
)
subject = vision[vision["image"].isin(sample_names)].sort_values("score", ascending=False).iloc[0]

image = Image.open(SAMPLE / "bathtub" / subject["image"]).convert("RGB")
box = [subject.x1, subject.y1, subject.x2, subject.y2]
crop = image.crop(pad_box(box, image.size, 0.35))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
axes[0].imshow(image)
axes[0].add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                            fill=False, edgecolor="#E07B39", linewidth=2.5))
axes[0].set_title("detection in the full image", fontsize=10)
axes[1].imshow(crop)
axes[1].set_title("what the model actually sees (35% padding)", fontsize=10)
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

print(f"{subject['image']}   score {subject['score']:.3f}")"""),

        ("md", """### Same detection, both approaches

This is the comparison the whole project builds toward."""),
        ("code", """text_row = generated[generated["image"] == subject["image"]].iloc[0]

print("TEXT-ONLY CONDITIONING")
print("=" * 78)
for col in question_cols:
    print(f"  {col.replace('generated_question_', ''):14s} {text_row[col]}")

print()
print("VISION CONDITIONING (Gemini 3.0 Pro Preview, same detection)")
print("=" * 78)
for perspective in ["single", "couple", "group"]:
    print(f"  travelling {perspective}:")
    for i in (1, 2, 3):
        print(f"      {subject[f'gemini_3_{perspective}_q{i}']}")"""),

        ("md", """The vision-conditioned questions are about *this* bathroom. They reference what
is in the frame. And because the model can see the scene, conditioning on who is
travelling produces genuinely different questions rather than the same sentence
with a word swapped."""),

        ("md", """### Measuring it

`scripts/build_finding3_table.py` regenerates this from the committed CSVs.

The comparison is **size-matched**: `distinct-n` falls as a corpus grows, so
comparing 2,484 vision questions against 1,003 text-only ones would flatter the
vision numbers. The vision row is the mean over 20 random subsamples of exactly
1,003 questions."""),
        ("code", """import random, statistics

vision_cols = [c for c in vision.columns if c.startswith("gemini_3_")]
vision_qs = [str(v).strip() for c in vision_cols for v in vision[c].dropna() if str(v).strip()]

d2, dup = [], []
for seed in range(20):
    random.seed(seed)
    stats = qm.summarize(random.sample(vision_qs, len(generated)))
    d2.append(stats["distinct_2"])
    dup.append(stats["duplicate_rate"])

comparison = quality[["distinct_2", "duplicate_rate"]].copy()
comparison["conditioning"] = "text"
comparison.loc["gemini_vision"] = [statistics.mean(d2), statistics.mean(dup), "vision"]
comparison"""),
        ("code", """text_best = quality["distinct_2"].max()
print(f"Best text-only distinct-2 : {text_best:.3f}")
print(f"Vision distinct-2         : {statistics.mean(d2):.3f}  "
      f"({statistics.mean(d2)/text_best:.0f}x better, range {min(d2):.3f}–{max(d2):.3f})")
print(f"Duplicate rate            : {quality['duplicate_rate'].min():.1%} → "
      f"{statistics.mean(dup):.1%}")"""),

        ("md", """## Two caveats I want to state myself

**This is not a clean ablation.** Gemini 3.0 Pro Preview is a stronger model than
Flan-T5 independently of what it is shown. The comparison establishes that
*vision-conditioning with a capable model* beats *text-conditioning with these
models*. It does not isolate the contribution of the crop. The clean experiment
is the same model run with and without the image, and it was not done.

**Diversity is not quality.** These questions are more varied. They were never
put in front of the study participants, so there is no evidence they are more
*helpful*. Notebook 04 shows why that distinction matters more than it sounds."""),
    ],

    # ------------------------------------------------------------------ 04
    "04_evaluation_and_results.ipynb": [
        ("md", """# 04 · Evaluation, and what the numbers actually mean

Three layers: automatic question metrics, backend reliability, and a
37-participant user study.

The study is where the project's assumptions got tested against real people, and
it is where the most uncomfortable finding lives."""),
        ("code", BOOTSTRAP),

        ("md", """## Choosing the vision model

Notebook 03 established that conditioning on the image beats conditioning on a
text description of it. That leaves the question of *which* multimodal model.

Three Gemini variants were compared on identical crops and identical prompts,
scored by BERTScore against 25 hand-written reference questions (5 per amenity)."""),
        ("code", """bert = pd.read_csv(
    RESULTS / "analysis_summary" / "model_comparison" / "bertscore_by_amenity.csv"
)
fallback_models = pd.read_csv(
    RESULTS / "analysis_summary" / "model_comparison" / "model_fallback_rates_summary.csv",
    encoding="utf-8-sig",
)

comparison = bert.pivot(index="model", columns="amenity", values="bertscore")
comparison["overall"] = comparison.mean(axis=1).round(3)
comparison.sort_values("overall", ascending=False)"""),
        ("code", """fallback_models"""),

        ("md", """![Model selection](../results/figures/generation/model_selection.png)

Gemini 3.0 Pro Preview wins on both measures, so it was used for everything
downstream. But the honest reading is that **the spread is under 2% on
BERTScore**, so model choice is nearly irrelevant at this scale, and the fallback
rate is the more useful discriminator. Reporting only the winner would have
implied a decisiveness the data does not support."""),

        ("md", """## Question quality

Two measures, both computed over the full Gemini 3.0 Pro Preview output.

**Semantic diversity** is measured *within* each image's question set: one minus
the mean pairwise cosine similarity between the questions generated for a single
detection. It asks whether the model varies its questions about one photo, a
different and more demanding question than whether it varies across photos."""),
        ("code", """diversity = pd.read_csv(RESULTS / "analysis_summary" / "semantic_diversity_by_amenity.csv")
fallback_amenity = pd.read_csv(
    RESULTS / "analysis_summary" / "model_comparison" / "gemini30_fallback_rates.csv",
    encoding="utf-8-sig",
)

display(diversity)
display(fallback_amenity)"""),

        ("md", """![Question quality by amenity](../results/figures/generation/generation_quality.png)

Mean semantic diversity is **0.709**, ranging from 0.628 for mirror to 0.809 for
hairdryer. The model is not asking the same thing five times about one
photograph.

The fallback pattern is the more interesting half. TV fails on **0.67%** of
questions; bathtub on **11.30%**, mirror on **10.88%**. Televisions are
standardised dark rectangles. Bathtubs vary in shape and framing; mirrors reflect
the rest of the room and are frequently ambiguous about where the object even
ends.

**Generation robustness tracks visual regularity.** That points at per-amenity
prompt work rather than a global model swap. The failure is concentrated, not
diffuse."""),

        ("md", """## Does traveller-profile conditioning actually work?

The pipeline conditions generation on who is travelling. Worth checking whether
the model attends to that or quietly ignores it, which is the usual outcome when
a conditioning signal is weak."""),
        ("code", """distinctiveness = pd.read_csv(
    RESULTS / "analysis_summary" / "profile_distinctiveness.csv",
    encoding="utf-8-sig",
)
distinctiveness.pivot(index="amenity", columns="pair", values="distinctiveness").round(3)"""),

        ("md", """![Profile conditioning](../results/figures/generation/profile_conditioning.png)

Every profile pair separates by **0.50 to 0.73** cosine distance. The
conditioning lands. Single-vs-group separates most on average (0.664), which is
the pair you would expect to diverge: a solo traveller and a family group want
genuinely different things from the same bathtub. It is worth noting the
exception, though. For bathtub and hairdryer, couple-vs-group separates more."""),

        ("md", """## The user study

37 participants rated 75 generated questions 1–5 on how helpful each would be
when choosing a hotel. Full protocol, demographics, and limitations:
[`docs/user_study_summary.md`](../docs/user_study_summary.md).

**Overall mean: 3.36 / 5.** That is a moderate result and is reported as one.
87% of questions cleared "moderately helpful"; only 37% reached 3.5+."""),
        ("code", """study = pd.DataFrame({
    "amenity": ["Bathtub", "Kettle", "Hairdryer", "TV", "Mirror"],
    "mean": [3.45, 3.45, 3.41, 3.28, 3.22],
    "worst question": [2.76, 3.05, 2.84, 2.19, 2.78],
    "best question": [3.86, 4.08, 3.73, 4.03, 3.62],
})
study["spread"] = (study["best question"] - study["worst question"]).round(2)
study"""),

        ("md", """## The finding that matters

Look at the `spread` column, not the `mean` column.

The gap *between* amenities is small, 3.22 to 3.45, which is noise at this
sample size. The gap *within* each amenity is large. TV ranges from 2.19 to 4.03
on the same amenity, from the same pipeline.

So the interesting question is not "which amenity works best" but "what separates
a good question from a bad one"."""),
        ("code", """best_worst = pd.DataFrame({
    "rating": [4.08, 4.03, 3.92, 2.78, 2.22, 2.19],
    "question is about": [
        "whether the appliance looked modern and clean",
        "whether the TV supported streaming apps",
        "whether guests could use their own streaming accounts",
        "whether a mirror was framed or frameless",
        "whether a white border was a display effect or a bezel",
        "whether a TV's screen edge was physical or rendered",
    ],
    "kind": [
        "booking decision", "booking decision", "booking decision",
        "visual detail", "visual ambiguity", "visual ambiguity",
    ],
})
best_worst"""),

        ("md", """**Participants rated questions about booking decisions highly, and questions
about visual ambiguity poorly.**

The low-scoring questions are exactly the ones a detector-driven system produces
naturally. When a model is uncertain whether that white border is a bezel or a
display artefact, the obvious move is to ask about it. Every participant rating
says: nobody cares.

**Detection uncertainty and traveller relevance are different quantities.** A
system that generates questions about whatever the model found ambiguous is
optimising the wrong objective, and it will feel broken to users while its
metrics look fine.

That reframes the whole pipeline. The useful signal is not "what is the detector
unsure about" but "what would change someone's booking decision", and those are
close to uncorrelated."""),

        ("md", """## Where the evidence is weak

Stated plainly, because it bounds everything above.

- **No control condition.** Participants never rated human-written or randomly
  selected questions. 3.36/5 has no reference point; it could be excellent or
  mediocre and this study cannot distinguish them. This is the single biggest
  weakness of the evaluation design.
- **Small, skewed sample.** 37 people, nearly 60% aged 18–25.
- **Stated preference, not behaviour.** Nobody booked anything. Rating a question
  as helpful is not evidence it helps.
- **Prompt and threshold come from the sweep itself**, so the headline figures
  are best-case rather than held-out.
- **A small part of the TV set is stock photography** rather than accommodation
  listings, so TV is slightly less domain-consistent than the other four.
- **The vision-conditioned questions were never rated at all.** The study covered
  text-only output, so the diversity gain in notebook 03 has no helpfulness
  evidence behind it."""),

        ("md", """## What I would do next

The ablation this project stops short of: the **same model, with and without the
cropped region**, everything else fixed. That separates conditioning from model
capability and would turn Finding 3 from suggestive into conclusive.

Then re-run the study with two additions: a human-written control condition, and
the vision-generated questions included. That gives the helpfulness scores a
reference point and tests whether diversity actually buys relevance.

And on the detection side: per-amenity thresholds calibrated against the cost of
a wrong question, rather than a single F1-maximising number. The user study
suggests a false positive is expensive in a way F1 does not capture."""),
    ],
}


def build(name: str, cells: list[tuple[str, str]]) -> Path:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(body) if kind == "md" else nbf.v4.new_code_cell(body)
        for kind, body in cells
    ]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }

    path = NB_DIR / name
    NB_DIR.mkdir(exist_ok=True)
    client = NotebookClient(
        nb, timeout=600, kernel_name="python3",
        resources={"metadata": {"path": str(NB_DIR)}},
    )
    client.execute()
    nbf.write(nb, path)
    return path


def main() -> int:
    for name, cells in NOTEBOOKS.items():
        path = build(name, cells)
        written = nbf.read(path, as_version=4)
        code = [c for c in written.cells if c.cell_type == "code"]
        with_output = [c for c in code if c.get("outputs")]
        print(f"built  {path.relative_to(REPO_ROOT)}  "
              f"({len(with_output)}/{len(code)} code cells produced output)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
