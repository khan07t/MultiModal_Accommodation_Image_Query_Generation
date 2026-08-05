# Dataset Inventory

## Provenance

The images are **real accommodation listing photographs, provided by trivago
N.V.** from their accommodation image inventory, and used for this research with
their permission.

One exception worth noting: roughly 5% of the TV test split is public stock
photography rather than trivago listings, including a few product shots and
outdoor billboards. Those images are not trivago's and are not covered by the
permission above.

That provenance matters for reading the results. This is not a general-purpose
object-detection benchmark reused for a hotel task. It is the actual domain the
system is intended for, with the clutter, lighting, and framing of real listing
photography.

How the dataset was built:

1. **Retrieval.** Images were extracted from trivago's image inventory, scoped to
   well-curated listings in three European cities (Paris, Barcelona, and Rome),
   chosen for accommodation density and for consistent availability of multiple
   room images per property.
2. **Filtering.** The inventory carries coarse room-level tags (bathroom,
   bedroom, kitchen, amenities) but no object-level labels, so target amenities
   could not be queried directly. Candidate images were retrieved by room tag and
   then **manually reviewed** one by one. Ambiguous, duplicated, corrupted, and
   irrelevant images were discarded.
3. **Metadata stripped.** Only image content was retained; all other metadata
   fields were discarded, so no textual or structured signal could leak into a
   pipeline that is supposed to work from pixels alone.
4. **Annotation.** Bounding boxes were drawn by hand in a **private** Roboflow
   workspace, one project per amenity to avoid class ambiguity, then exported in
   COCO format.

Critically: **no annotated data was used to train or fine-tune the detector.**
OWL-ViT is used strictly zero-shot. The annotations exist only to evaluate it.

A 46-image sample is committed under [`data/sample/`](../data/sample/) so the
notebooks run on a clean checkout. The full splits are not committed.

## Dataset composition

Per amenity, as reported in the thesis (Table 3.1). "Negative" images come from
**similar room contexts**, a bathroom with no bathtub rather than an unrelated
scene, which is what makes the false-positive analysis meaningful.

| Amenity | Total images | Positive | Negative |
| --- | ---: | ---: | ---: |
| Bathtub | 292 | 216 | 76 |
| Kettle | 273 | 201 | 72 |
| TV | 338 | 247 | 91 |
| Hairdryer | 290 | 215 | 75 |
| Mirror | 378 | 285 | 93 |

**This table and the split table below do not reconcile, and that is expected.**
Two things differ:

1. **They count different objects.** "Positive" here is the *annotation* count,
   not an image count. One photograph can hold several instances of an amenity,
   so mirror's 285 boxes are spread across fewer than 285 images.
2. **They were measured at different points.** The composition table is the
   dataset as described in the thesis. The split table is what the committed
   COCO exports actually contain, counted directly off disk. Annotation passes
   continued after the thesis figures were fixed, so TV shows 338 against 360 and
   mirror 378 against 325.

Where the two disagree, **the split table governs**, because every reported
metric is computed from those exports. The composition table is kept for
traceability against the thesis.

## Splits

| Dataset | Split | Images on disk | COCO images | Annotations |
| --- | ---: | ---: | ---: | ---: |
| Bathtub | train | 411 | 411 | 415 |
| Bathtub | valid | 18 | 18 | 19 |
| Bathtub | test | 290 | 290 | 216 |
| Hairdryer | train | 367 | 367 | 412 |
| Hairdryer | valid | 91 | 91 | 104 |
| Hairdryer | test | 284 | 284 | 215 |
| Kettle | train | 524 | 527 | 698 |
| Kettle | valid | 99 | 99 | 145 |
| Kettle | test | 272 | 272 | 201 |
| Mirror | train | 261 | 261 | 267 |
| Mirror | valid | 52 | 52 | 53 |
| Mirror | test | 325 | 325 | 285 |
| TV | train | 201 | 201 | 201 |
| TV | valid | 53 | 53 | 53 |
| TV | test | 360 | 360 | 274 |

All reported metrics come from the **test** splits. Since the detector is never
trained, the train and valid splits play no part in the reported numbers.

## Known data-quality issues

Recorded rather than quietly worked around, because they bound how much the
headline numbers should be trusted:

1. **Validation splits are small.** Bathtub has 18 validation images, mirror 52.
   Too small for confident model selection, which is why threshold choices are
   reported as a full sweep rather than a single tuned value.
2. **Three missing image files.** `Kettle/train/_annotations.coco.json`
   references three images absent from the export. Training split only; no effect
   on reported test metrics.
3. **Inconsistent category naming across exports.** Each export shipped different
   category labels (`Bathub-Bathtub-Object`, `bathtub`, `bathtub_eval`). A `none`
   category appears in several exports and is stripped during preparation; images
   labelled `none` are retained as the negative set.
4. **Annotation tightness varies.** Boxes were drawn by hand across sessions.
   Some hug the object, others include surrounding fixtures. This caps achievable
   IoU independently of detector quality, and is part of why mean IoU
   (0.573–0.750) sits well below F1.
5. **Geographic and listing-quality concentration.** Three European cities, and
   well-curated listings rather than a random sample of the inventory.
   Performance on lower-quality photography, or in other markets, is untested.
6. **A small amount of stock photography in the TV split.** About 5% of TV
   images are public stock rather than accommodation listings, so TV is slightly
   less domain-consistent than the other four amenities.
7. **Prompt and threshold were selected on the test split.** The validation
   splits were too small to select on, so the reported best-per-amenity figures
   are best-of-sweep on test and are optimistic. The full sweep is published so
   the effect is visible rather than hidden.

## Expected layout

```
data/
  sample/                      committed: 46 images, runs the notebooks
  bathtub/
    train/_annotations.coco.json + images
    valid/_annotations.coco.json + images
    test/_annotations.coco.json  + images
  hairdryer/
  kettle/
  mirror/
  tv/
```

Point the pipeline at the full data with `--data-root data/`.

## Attribution

Accommodation images provided by **trivago N.V.** and reproduced with
permission. They remain the property of trivago N.V. and are not covered by this
repository's MIT licence, which applies to the code only.
