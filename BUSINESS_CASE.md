# Business case

## The user problem

Choosing a hotel on a metasearch site means comparing many listings quickly. Photos carry a lot of the decision (is there a real bathtub, can I make tea in the room, will the TV stream), but that information stays implicit. Users have to spot it themselves and work out what to ask, and the questions depend on who is travelling. A family worries about capacity and safety, a couple about sharing the space, a solo business traveller about convenience.

When a detail goes unnoticed, the user either books with a doubt, drops off to check elsewhere, or keeps scrolling. All three cost trivago: a less confident clickout, a lost session, or more effort per decision.

## The idea

Use the photos that are already on the page to **suggest the right questions** for this traveller:

- detect which amenities are visible in a listing's photos
- generate short, practical questions about them
- tailor the questions to the traveller type the user already chose in search

The feature does not answer the questions or claim facts. It points attention to what matters, so the user checks it before clicking out. That keeps the risk low: a question cannot be "wrong" the way a generated fact can.

## Where it sits in the product

| Search page signal | Profile | Example question (kettle) |
|---|---|---|
| 1 adult | Solo | Is this stainless steel kettle efficient enough to boil just one cup of water quickly for my early morning coffee? |
| 2 adults, or "couple" trip type | Couple | Is the boiling mechanism quiet enough that I can make tea early without waking my partner in the same room? |
| 3+ guests, children, or "group" trip type | Group / family | Since the body is stainless steel, does the exterior get dangerously hot to the touch, which might be a safety concern for our children? |

Possible placements:
- next to the photo gallery on the listing ("Worth checking for your trip")
- as chips under an amenity photo
- as input to review search or a Q&A layer later on

Generation runs offline per photo and profile, so nothing slows down the page. Questions are pre-computed and cached against the image ID.

## Evidence so far

This was a research project. Nothing was tested live, so the evidence comes from offline metrics and a user study with 37 participants, recruited through university groups and forums, trivago forums, acquaintances and the general public.

| Signal | Result |
|---|---|
| Perceived usefulness of the questions | 3.36 / 5 overall. 49.6% of ratings were 4 or 5 |
| Would like the feature on a booking site | 67.6% moderately or extremely useful |
| AI questions vs typical hotel descriptions | 43.2% rated them more useful (4 or 5 of 5) |
| Most helpful amenity | Bathtub (37.8% of participants), then TV (24.3%) |
| Unprompted feedback | One participant asked for questions tailored to user profile and booking type, which the profile mode delivers |

## How I would measure it live

An A/B test on listing pages, with the question module shown to the treatment group.

**Primary KPI**
- **Clickout rate** per listing view: does the user move on to a booking partner more often?

**Secondary KPIs**
- **Engagement with the module:** share of listing views where questions are seen, expanded or clicked
- **Time to decision:** time from search to first clickout
- **Return-to-search rate:** fewer users bouncing back to the results list from a listing
- **Filter usage:** whether users refine by amenity after seeing a question

**Guardrails**
- Page load time: no regression, since everything is pre-computed
- Share of questions flagged by users as irrelevant or wrong
- Quality gate: fallback rate per amenity, as tracked offline (7.85% general, 0 to 5% per profile)

**Offline gates before any test**
- Detection precision per amenity above an agreed threshold (TV, bathtub, hairdryer and kettle qualify now; mirror needs work)
- Human spot checks on a sample of generated questions per amenity

## Rollout path

1. Start with the most reliable amenities: TV, bathtub, kettle, hairdryer. Hold back mirror until detection improves.
2. Pre-compute questions for the Tier A/B inventory in a few cities and run the A/B test there.
3. Add ranking: show only the questions the photo supports best. The study showed that grounded questions rate highest.
4. Extend to more amenities and languages, and connect to amenity metadata and reviews so questions can be answered as well as asked.

## Risks and how they are handled

| Risk | Mitigation |
|---|---|
| Detector finds an amenity that isn't there | Per-amenity thresholds tuned on negative images; mirror held back |
| Generic or obvious questions | Rule-based quality gate with amenity-specific fallbacks |
| Questions that need guesswork ("Is this a Frame TV?") | Rank by visual grounding; these were the lowest-rated in the study |
| Cost of large models | Offline batch generation, cached per image and profile |
| Scale beyond five amenities | Open-vocabulary detection: a new amenity needs a prompt, not a new model |
