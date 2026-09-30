-- Quick checks on the candidate list before downloading.

-- 1. Candidates per city and tag
SELECT city, tag_name, COUNT(*) AS images
FROM `project.scratch.candidate_images`
GROUP BY city, tag_name
ORDER BY city, images DESC;

-- 2. Missing or broken URLs
SELECT COUNT(*) AS missing_urls
FROM `project.scratch.candidate_images`
WHERE image_url IS NULL OR image_url = '';

-- 3. Properties that would dominate the sample
SELECT property_id, COUNT(*) AS images
FROM `project.scratch.candidate_images`
GROUP BY property_id
HAVING COUNT(*) > 20
ORDER BY images DESC;

-- 4. Random sample to eyeball before the manual review
SELECT image_id, image_url, tag_name, city
FROM `project.scratch.candidate_images`
WHERE RAND() < 0.01
LIMIT 50;
