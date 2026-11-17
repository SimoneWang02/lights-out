-- Example ad-hoc queries over the Hive tables registered by the pipeline (hive: true).
-- Replace `lightsout` with the database name from conf/dataproc.yaml.
-- The same statements run in spark-sql; locally, read the Parquet folders under data/ instead.

USE lightsout;

-- 1. Day vs night pedestrian crashes by year.
SELECT year(crash_date)                         AS yr,
       period,
       count(*)                                 AS crashes,
       sum(ped_injured)                         AS pedestrians_injured,
       sum(ped_killed)                          AS pedestrians_killed
FROM gold_crashes
WHERE is_ped
GROUP BY year(crash_date), period
ORDER BY yr, period;

-- 2. Pedestrian crashes per hour of darkness vs per hour of daylight, by month.
--    Night hours carry more risk per hour in winter, when the evening rush is dark.
SELECT c.mon, c.period, c.crashes, p.hours, round(c.crashes / p.hours * 1000, 2) AS crashes_per_1000_hours
FROM (SELECT month(period_date) AS mon, period, count(*) AS crashes
      FROM gold_crashes WHERE is_ped GROUP BY month(period_date), period) c
JOIN (SELECT month(`date`) AS mon, period, sum(hours) AS hours
      FROM gold_periods GROUP BY month(`date`), period) p
  ON c.mon = p.mon AND c.period = p.period
ORDER BY c.mon, c.period;

-- 3. How long do dark-lamp complaints stay open? (shows why closed_date needs care)
SELECT CASE WHEN interval_quality = 'negative' THEN '0 closed before created'
            WHEN interval_quality = 'open'     THEN '9 still open'
            WHEN duration_days < 1             THEN '1 under 1 day'
            WHEN duration_days < 7             THEN '2 1-7 days'
            WHEN duration_days < 30            THEN '3 7-30 days'
            WHEN duration_days < 90            THEN '4 30-90 days'
            ELSE                                    '5 90 days or more' END AS bucket,
       count(*) AS complaints
FROM gold_complaints
WHERE kind = 'dark'
GROUP BY 1
ORDER BY 1;

-- 4. Night pedestrian crashes that happened in a cell while a dark-lamp complaint was open
--    in its footprint (main definition: closed-date window, cell plus neighbours).
SELECT c.borough, count(*) AS night_ped_crashes_during_outage
FROM gold_crashes c
JOIN gold_exposure e
  ON  e.cell = c.cell AND e.`date` = c.period_date AND e.period = c.period
WHERE c.is_ped AND c.period = 'night'
  AND e.spec = 'A' AND e.geo = 'r10k1' AND e.kind = 'dark'
GROUP BY c.borough
ORDER BY night_ped_crashes_during_outage DESC;

-- 5. Cells with the most outage nights since 2020 (same cell only).
SELECT cell, count(*) AS outage_nights, max(n_complaints) AS most_complaints_open_at_once
FROM gold_exposure
WHERE spec = 'A' AND geo = 'r10k0' AND kind = 'dark' AND period = 'night'
GROUP BY cell
ORDER BY outage_nights DESC
LIMIT 20;

-- 6. Open complaints ranked by priority score vs by arrival.
SELECT rank_by_score, rank_by_arrival, round(score, 2) AS score, days_open, borough,
       intersection_street_1, intersection_street_2
FROM out_open_ranked
ORDER BY rank_by_score
LIMIT 25;

-- 7. Effect estimates table produced by stage 5.
SELECT label, round(rr_night, 3) AS rr_night, round(rr_day, 3) AS rr_day,
       round(ratio, 3) AS ratio, round(ratio_lo, 3) AS lo, round(ratio_hi, 3) AS hi,
       night_crashes_outage
FROM out_effects;
