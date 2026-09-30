-- Candidate images for one amenity.
-- Change @room_tags per amenity, e.g. ['bathroom'] for bathtub / hairdryer,
-- ['bedroom'] for TV, ['bedroom', 'kitchen'] for kettle.

DECLARE room_tags ARRAY<STRING> DEFAULT ['bathroom'];

WITH properties AS (
  SELECT property_id, city, property_tier
  FROM `project.inventory.properties`
  WHERE city IN ('Paris', 'Barcelona', 'Rome')
    AND property_tier IN ('A', 'B')
),

tagged_images AS (
  SELECT
    i.image_id,
    i.property_id,
    i.image_url,
    i.width,
    i.height,
    t.tag_name
  FROM `project.inventory.accommodation_images` AS i
  JOIN `project.inventory.image_tags` AS t
    ON i.image_id = t.image_id
  WHERE i.is_active
    AND t.tag_name IN UNNEST(room_tags)
    AND i.width >= 640            -- skip low-resolution photos
)

SELECT
  ti.image_id,
  ti.image_url,
  ti.tag_name,
  p.city,
  p.property_tier
FROM tagged_images AS ti
JOIN properties AS p
  ON ti.property_id = p.property_id
QUALIFY ROW_NUMBER() OVER (PARTITION BY ti.image_url ORDER BY ti.image_id) = 1   -- same photo reused across listings
ORDER BY p.city, ti.property_id;
