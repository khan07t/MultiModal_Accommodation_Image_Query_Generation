# BigQuery queries

These show how the candidate images were pulled from trivago's image inventory. Table and column names are generalised, since the real tables are internal.

The inventory has no object-level labels, only room-level tags. So the queries narrow the pool down to the right rooms, and the final filtering was done by eye.

| File | Purpose |
|---|---|
| `01_explore_tags.sql` | What tags exist and how many images each has, per city |
| `02_candidate_images.sql` | Candidate images per amenity: join images with properties, keep Tier A/B in Paris, Barcelona, Rome, filter by room tag |
| `03_checks.sql` | Sanity checks before download: duplicates, missing URLs, images per property |

| Amenity | Room tags |
|---|---|
| Bathtub | bathroom |
| Hairdryer | bathroom |
| Mirror | bathroom, bedroom |
| TV | bedroom |
| Kettle | bedroom, kitchen |
