"""User study: load the questionnaire export and summarise ratings."""

import pandas as pd

# Column layout of the Google Forms export: 7 intro columns, then 15 rating
# columns per amenity (3 images x 5 questions), then 4 closing questions.
AMENITY_COLUMNS = {
    "bathtub": (7, 22),
    "kettle": (22, 37),
    "tv": (37, 52),
    "hairdryer": (52, 67),
    "mirror": (67, 82),
}
INTRO = {3: "age_range", 4: "travel_frequency", 5: "booked_online_2y", 6: "photo_importance",
         1: "warmup_rating"}
CLOSING = {82: "feature_useful", 83: "vs_descriptions", 84: "most_helpful_amenity", 85: "comment"}


def split_export(raw: pd.DataFrame):
    """Return (participants, ratings_long) without timestamps."""
    raw = raw.reset_index(drop=True)
    ids = [f"P{i + 1:02d}" for i in range(len(raw))]

    participants = pd.DataFrame({"respondent_id": ids})
    for idx, name in {**INTRO, **CLOSING}.items():
        participants[name] = raw.iloc[:, idx].values

    rows = []
    for amenity, (a, b) in AMENITY_COLUMNS.items():
        for k, col in enumerate(raw.columns[a:b]):
            question = col.split(":", 1)[1].strip() if ":" in col else col
            for rid, rating in zip(ids, raw[col]):
                rows.append(dict(respondent_id=rid, amenity=amenity, image=k // 5 + 1,
                                 question_no=k % 5 + 1, question=question, rating=rating))
    return participants, pd.DataFrame(rows)


def question_summary(ratings):
    g = ratings.groupby(["amenity", "image", "question_no", "question"])["rating"]
    return g.agg(mean="mean", std="std", n="count").reset_index()


def amenity_summary(ratings):
    g = ratings.groupby("amenity")["rating"]
    out = g.agg(mean="mean", std="std", n="count").reset_index()
    out["share_4_or_5"] = ratings.assign(hi=ratings.rating >= 4).groupby("amenity")["hi"].mean().values
    return out.sort_values("mean", ascending=False)
