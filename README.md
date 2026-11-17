# Lights Out

Streetlight outages and nighttime pedestrian collisions in New York City.
CS-GY 6513 Big Data, Fall 2026 — Yiyuchen Hu, Steven Kang, Simone Wang.

The pipeline links NYPD Motor Vehicle Collisions to 311 Street Light Condition complaints by
place (H3 hexagons) and time (sunrise/sunset), estimates how crash risk changes at night while
a complaint is open, and ranks open complaints by the nighttime pedestrian risk around them.

## Run it

Requirements: [uv](https://docs.astral.sh/uv/) and a Java 17 runtime (`JAVA_HOME`, or a JDK
under `~/.local/jdk/current`).

```bash
uv sync                 # Python 3.11 environment with PySpark 3.5
uv run pytest           # unit tests, including the planted-effect test
./run_all.sh            # all eight stages; ./run_all.sh 5 reruns from stage 5
```

| Stage | Job | Reads | Writes |
|---|---|---|---|
| 1 | `jobs/01_ingest.py` | NYC Open Data API | `data/raw/` yearly CSVs |
| 2 | `jobs/02_clean.py` | raw CSV | `silver/collisions`, `silver/complaints`, `out/quality` |
| 3 | `jobs/03_keys.py` | silver | `gold/sun`, `gold/periods`, `gold/crashes`, `gold/complaints` |
| 4 | `jobs/04_exposure.py` | gold | `gold/exposure` |
| 5 | `jobs/05_effect.py` | gold | `out/effects`, `out/event_study` |
| 6 | `jobs/06_score.py` | gold | `out/cell_scores`, `out/score_curve`, `out/open_ranked` |
| 7 | `jobs/07_ml.py` | gold, `out/cell_scores` | `gold/ml_dataset`, `out/ml_metrics*`, `out/ml_predictions*` |
| 8 | `jobs/08_viz.py` | gold, out | `reports/figures/*.png`, `reports/tables/*`, `reports/map.html` |

All tables are Parquet under `data_root` from the config file. Every job takes
`--conf conf/<name>.yaml`; analysis parameters (H3 resolution, window lengths, training cut-off)
live there.

## How the analysis works

**Time.** Sunrise and sunset for each date come from `astral` and are stored as a small lookup
table. A crash is "night" if it happens before sunrise or after sunset; a crash at 01:30
belongs to the previous evening's night. All timestamps are NYC wall-clock time.

**Space.** Each crash and complaint gets an H3 resolution-10 cell (hexagon edge about 66 m).
A complaint's footprint is its cell plus the six neighbours.

**Outage windows.** `closed_date` is not a reliable repair time (see Data caveats), so two
definitions run side by side:

- **A** — created to closed, capped at 90 days, rows with closed-before-created excluded;
- **B** — a fixed 7 days after the report, ignoring `closed_date` entirely.

**Effect estimate.** Each (cell, month) is its own control. Within it, night hours are split
into outage and non-outage, and crash rates are compared with a Mantel–Haenszel rate ratio.
The same ratio is computed for daytime crashes; a dead lamp cannot matter in daylight, so the
day ratio captures everything else that differs while a complaint is open. The headline number
is **night ratio ÷ day ratio**. Intervals come from a bootstrap that resamples neighbourhoods
(H3 resolution-8 cells).

**Checks on the estimate.**

- *Placebo complaints* — door open, wire exposed, lamp burning by day: the lamp still works,
  so these should show no night effect.
- *Event study* — crash rates in the footprint for the 28 nights before and after a report.
- *Sensitivity grid* — window definition, footprint size, cell resolution, twilight excluded,
  other crash outcomes.
- *Planted-effect test* (`tests/test_effect.py`) — a synthetic city with a known effect of 1.5
  and deliberate confounding by place and season; the estimator recovers 1.5, and recovers 1.0
  when no effect is planted.

**Priority score.** Nighttime pedestrian crashes in a complaint's footprint during the training
period (2020–2024), shrunk toward the borough average with a gamma–Poisson model because most
footprints have zero or one crash. Tested on complaints from 2025 on: share of nearby crashes
reached by the top k% of complaints by score, against arrival order.

**Classifier.** Spark MLlib logistic regression and gradient-boosted trees predicting a
nighttime pedestrian crash in the footprint within 7 nights of the report, trained on
2021–2024 and tested on 2025 onward, compared with the priority score as a baseline.

## Results

From the run of 2026-11-17, covering 2020-01-01 to 2026-11-13. Figures are in
`reports/figures/`, the numbers behind them in `reports/tables/`, the map in `reports/map.html`.

**1. No measurable rise in crash risk while a streetlight complaint is open.**

| Analysis | Night ÷ day ratio | 95% interval | Night crashes during outage |
|---|---|---|---|
| Main (closed-date window, 7-cell footprint), pedestrian | 0.95 | 0.87 – 1.04 | 4,499 |
| Fixed 7-day window, pedestrian | 1.02 | 0.95 – 1.09 | 2,391 |
| Closed-date window capped at 30 days, pedestrian | 1.00 | 0.92 – 1.09 | 2,695 |
| Any injury crash | 0.99 | 0.95 – 1.03 | 19,715 |
| All crashes | 1.02 | 0.99 – 1.04 | 47,330 |
| Placebo complaints, closed-date window | 1.02 | 0.80 – 1.30 | 411 |
| Placebo complaints, fixed window | 0.99 | 0.84 – 1.16 | 342 |

Every variant sits at 1.0 within its interval. For pedestrian crashes the intervals exclude an
increase above roughly 4–9%, depending on the window definition. This is a statement about
*open complaints*; because reports lag the true outage, a small real effect of a dark lamp
cannot be excluded.

The null is not a pipeline failure: duplicating half of the night pedestrian crashes inside
outage windows in the real data (a planted effect of 1.5) moves the main ratio from 0.95 to
1.44, against 1.43 expected.

The event study agrees: in the week before a report the night ÷ day ratio is 1.05
(0.98 – 1.13) relative to the earlier baseline, and 1.03 (0.96 – 1.11) in the week after.

**2. Darkness itself clearly matters.** Between 17:00 and 19:00 there are 524 pedestrian
crashes per 100 days in November–January, when those hours are dark, against 270 in May–July,
when they are light. Midday (12:00–14:00) changes far less: 239 against 210.

**3. Ranking by score beats arrival order by a wide margin.** On 57,343 complaints from 2025
onward (740 night pedestrian crashes within 7 nights nearby):

| Complaints handled first | Crashes covered, by score | By arrival order |
|---|---|---|
| 10% | 29% | 10% |
| 20% | 49% | 20% |
| 50% | 82% | 51% |

The gain comes from where crashes concentrate, not from an outage effect: the score sends
crews first to places where pedestrians are hit at night anyway.

**4. The classifier adds little over the score for pedestrian crashes.** Top-decile lift on
the test period: logistic regression 3.1, gradient-boosted trees 3.1, priority score 2.9,
last year's count alone 2.4 (base rate 1.3%). For the broader target "any night crash within
7 nights" (base rate 10.7%) the models help more: PR area 0.26 against 0.21 for the score.

## Data caveats

- About 14% of dark-lamp complaints have a closed date *before* the created date and about a
  third close within a day; the resolution text is one generic sentence. Hence the two window
  definitions.
- About 18% of complaints and 7% of crashes have no usable coordinates and are excluded.
  Most of those complaints do carry street names, so they could be recovered by geocoding.
- A complaint marks when someone *reported* a lamp, not when it failed. Unreported dark nights
  count as "no outage", which pulls the estimate toward no effect.
- Crashes are police-reported only.

## Running on NYU Dataproc

1. Copy the raw files: `hdfs dfs -put data/raw /user/<netid>/lightsout/raw`.
2. Edit `conf/dataproc.yaml`: set `data_root` and `spark.database` to your NetID.
3. `astral` is needed on the driver only. `h3` is needed on executors for stages 3–7. Either
   ship a packed environment (`--archives env.tar.gz#environment` with
   `spark.pyspark.python=./environment/bin/python`), or pass `--h3-mode driver` to stages 3–7,
   which computes cells on the driver and needs `h3` only there.
4. Submit each job with `spark-submit --py-files lightsout.zip jobs/0N_*.py --conf conf/dataproc.yaml`
   (`cd src && zip -r ../lightsout.zip lightsout`).
5. With `hive: true`, every table is also registered as `<database>.<layer>_<name>`;
   `hive/queries.sql` has example queries.
