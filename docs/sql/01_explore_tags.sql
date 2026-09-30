-- How are images tagged, and how many do we have per city?
-- Table and column names generalised.

SELECT
  p.city,
  t.tag_name,
  COUNT(DISTINCT i.image_id)  AS images,
  COUNT(DISTINCT i.property_id) AS properties
FROM `project.inventory.accommodation_images` AS i
JOIN `project.inventory.image_tags` AS t
  ON i.image_id = t.image_id
JOIN `project.inventory.properties` AS p
  ON i.property_id = p.property_id
WHERE p.city IN ('Paris', 'Barcelona', 'Rome')
  AND p.property_tier IN ('A', 'B')
  AND i.is_active
GROUP BY p.city, t.tag_name
ORDER BY p.city, images DESC;
