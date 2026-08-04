# User Study Summary

Human evaluation of the generated amenity questions, run in December 2025.

## Design

| | |
| --- | --- |
| Participants | 37 |
| Questions rated | 75 (5 per image × 15 images) |
| Amenities | Bathtub, Kettle, TV, Hairdryer, Mirror |
| Scale | 1 (not helpful) – 5 (extremely helpful) |
| Duration | 12–15 minutes |
| Consent | 100% of participants consented |

Participants were shown an accommodation photo and five AI-generated questions
about the detected amenity, and rated **how helpful each question would be when
deciding whether to book**, explicitly *not* whether the question could be
answered from the photo.

### Participant profile

- **Age:** 59.5% aged 18–25, 27% aged 26–35, 13.5% aged 36+
- **Travel frequency:** 64.9% travel 2–4 times per year, 18.9% travel 5–10 times
- **Online booking experience:** 86.5% had booked accommodation online in the past two years
- **Importance of photos to booking:** 59.4% rated photos "very important" or "critical"

Responses were collected anonymously; no personally identifying information was
gathered.

## Results

**Overall mean helpfulness: 3.36 / 5** across all 75 questions.

| Amenity | Mean | Lowest-rated question | Highest-rated question |
| --- | ---: | ---: | ---: |
| Bathtub | 3.45 | 2.76 | 3.86 |
| Kettle | 3.45 | 3.05 | 4.08 |
| Hairdryer | 3.41 | 2.84 | 3.73 |
| TV | 3.28 | 2.19 | 4.03 |
| Mirror | 3.22 | 2.78 | 3.62 |

- 87% of questions scored ≥ 3.0 ("moderately helpful" or better)
- 37% of questions scored ≥ 3.5
- No amenity averaged below 3.2, and none averaged above 3.5

## What the ratings actually show

The between-amenity differences are small. The interesting variance is *within*
amenities. The spread between the best and worst question for TV was 1.84
points, far larger than any gap between categories.

**Highest-rated questions** asked about concrete, decision-relevant attributes:

| Rating | Question |
| ---: | --- |
| 4.08 | Whether the appliance looked modern and in clean condition |
| 4.03 | Whether the TV supported streaming apps |
| 3.92 | Whether guests could log into their own streaming accounts |
| 3.86 | Whether a separate walk-in shower existed alongside the tub |

**Lowest-rated questions** asked about visual ambiguity or niche features:

| Rating | Question |
| ---: | --- |
| 2.19 | Whether a TV was a "Frame"-style model displaying artwork |
| 2.22 | Whether a white border was a display effect or physical bezel |
| 2.76 | Whether bathroom tiles looked modern or somewhat dated |
| 2.78 | Whether a backlit border gave shadow-free illumination |

### The main finding

Participants rewarded questions about **things that would change a booking
decision** and penalised questions about **things the model happened to be
uncertain about**.

This is a meaningful negative result for the naive version of the system. The
obvious design, generating questions about whatever the detector found ambiguous,
optimises for the wrong thing. Detection uncertainty and traveller relevance are
different quantities, and the system should be driven by the second.

## Limitations

- **Small, skewed sample.** 37 participants, nearly 60% aged 18–25. Not
  representative of the broader travel-booking population.
- **No control condition.** Participants did not rate human-written questions or
  a random baseline, so 3.36 has no reference point. This is the single biggest
  weakness of the study design.
- **Single rating per question per participant**, no inter-rater reliability
  measure.
- **Stated preference, not behaviour.** Rating a question as helpful is not the
  same as engaging with it in a real booking flow.

---

*Raw response exports and the original study instrument are retained privately
and are available on request.*
