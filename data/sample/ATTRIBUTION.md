# Image attribution

## Images provided by trivago N.V.

The 46 photographs in this folder are real accommodation listing
images from trivago's image inventory, reproduced with permission for this
research. They remain the property of trivago N.V.

They are **not** covered by this repository's MIT licence, which applies to
the code only. If you fork this repository, the code is yours to reuse; the
images are not.

Bounding boxes were annotated by hand in a private Roboflow workspace and
exported in COCO format. No annotated data was used to train or fine-tune any
model. OWL-ViT runs zero-shot, and these annotations exist purely to
evaluate it.

| Amenity | Images | Ground-truth boxes |
|:---|---:|---:|
| bathtub | 10 | 8 |
| hairdryer | 10 | 9 |
| kettle | 10 | 8 |
| mirror | 8 | 9 |
| tv | 8 | 8 |

Boxes can outnumber images: a single photograph often contains several
instances of an amenity, which is why the two columns do not match.

Images carrying no box are the **negative set**: the same kind of room
without the target amenity, used to measure false positives.

This is a deterministic slice of the test splits, committed so the notebooks
run on a clean checkout. Full splits are not committed. See
[`docs/dataset_inventory.md`](../../docs/dataset_inventory.md) for provenance,
counts, and known data-quality issues.

Regenerate with `python scripts/build_sample_dataset.py --source <exports>`.
