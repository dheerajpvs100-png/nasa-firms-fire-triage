# NASA FIRMS Fire Hotspot Triage Dashboard

A Streamlit dashboard that takes satellite active-fire detections from NASA FIRMS (MODIS Terra/Aqua and VIIRS S-NPP, NOAA-20, NOAA-21) and ranks them by how much attention they probably deserve. It also tells you *why* a hotspot ranks where it does, because a bare number isn't much use to someone deciding what to look at first.

## Why I rebuilt this

The first version was a short script that sorted hotspots with a simple weighted sum and printed a table. It worked on the happy path and fell over almost everywhere else:

| Area | What the old script did | What this version does |
| :--- | :--- | :--- |
| Loading data | Broke on uppercase headers like `LATITUDE`, choked on VIIRS confidence letters (`l`, `n`, `h`), and had no idea that `bright_ti4` and `brightness` are the same thing. | Normalizes column names regardless of case, maps common synonyms (`bright_ti4`, `frp`, `scan`), drops bad rows, and converts letter confidences to numbers. |
| Scoring | Added raw values together with no calibration (see below). | A 0-100 urgency index built from log-scaled FRP, thermal excess over ambient, persistence, and confidence, with adjustable weights. |
| Map | There wasn't one, just a table. | An interactive PyDeck map: color by urgency tier, marker size by FRP, an optional heat layer, and hover tooltips. |
| Explanations | One hard-coded string with raw numbers dropped in. | Plain-English reasons for each ranking, plus the four subscores so you can see what pushed it up or down. Likely industrial heat sources get an advisory. |
| Testing | None. | A small labelled benchmark (five archetypes), a rank-correlation check against the old formula, and tests that scores stay inside 0-100. |
| Exports | CSV only. | Filtered CSV, GeoJSON for GIS tools, and an auto-generated incident briefing in Markdown. |
| Errors | `st.stop()` with no explanation. | Messages that say what went wrong, and a built-in California wildfire sample so you can try it without uploading anything. |

## How the scoring works

### What was wrong with the old formula

```
priority = 0.35 * brightness + 0.25 * confidence + 0.25 * FRP + 15 * repeats
```

Two problems made it unreliable:

- **Kelvin baseline.** Brightness temperature is in Kelvin and the ground sits around 300 K even when nothing is burning. So 0.35 x 300 = 105 points were handed out for free to every pixel, which swamped everything else.
- **Repeats dominated.** A 38 MW gas flare seen on 14 passes picked up 15 x 14 = 210 points from repetition alone and scored 362.75. A 320 MW crown fire scored 353.88. The flare ranked first. That's backwards.

### The new index

Four subscores, each on a 0-100 scale:

1. **Radiative intensity.** Fire radiative power spans a huge range (roughly 1 to 250+ MW), so it's log-scaled:

   S_frp = min(100, ln(1 + FRP) / ln(1 + 250) x 100)

2. **Thermal excess.** How far brightness temperature sits above the ~300 K ambient level, capped at 100:

   S_thermal = min(100, max(0, T_bright - 300))

3. **Persistence.** How many separate satellite overpasses saw it:

   S_persist = min(100, max(15, (N_repeats - 1) x 20 + 15))

4. **Confidence.** The detection algorithm's own confidence, 0-100.

The final score is a weighted sum:

```
score = w_frp * S_frp + w_persist * S_persist + w_thermal * S_thermal + w_conf * S_conf
```

The weights are adjustable in the sidebar.

### Tiers

| Tier | Score | Rough meaning |
| :--- | :--- | :--- |
| Critical | 75 and up | Very high heat output, or a fire that keeps showing up across passes. Look at these first. |
| High | 50-74 | Real, sustained burning. |
| Moderate | 30-49 | Something developing: an ag burn, a small local hotspot. |
| Low / Monitoring | under 30 | A single detection or low energy. Keep an eye on it. |

## Does it actually rank things sensibly?

There's an **Evaluation Benchmark** tab that runs five hand-built reference cases, each with a "ground truth" urgency I assigned by judgment:

| Case | Archetype | FRP (MW) | Passes | Conf | New score | New tier | Old score | Truth |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| TC-01 | Major wildfire | 320.0 | 7 | 98% | 99.7 | Critical | 353.88 | 100 |
| TC-02 | Fast-growing spot fire | 85.0 | 3 | 85% | 69.1 | High | 212.80 | 75 |
| TC-03 | Persistent industrial gas flare | 38.0 | 14 | 90% | 75.7* | Advisory | 362.75 (ranked #1) | 45 |
| TC-04 | Prescribed burn | 14.5 | 1 | 70% | 36.8 | Moderate | 148.82 | 25 |
| TC-05 | Solar glint / marginal anomaly | 2.1 | 1 | 32% | 18.1 | Low | 131.39 | 10 |

\* The dashboard flags this one with a stationary industrial heat source note in its explanation, so it isn't read as a wildfire.

What the checks show:

- Spearman rank correlation with the ground truth is **0.90** for the new index and **0.70** for the old formula.
- Every score lands inside 0-100, and nothing comes back null or NaN.
- All of the severe cases end up in the top tier.

Keep in mind that five cases is a sanity check, not a proper validation. I'd want real labelled incidents before trusting the weights in any serious setting.

## Running it

Start the dashboard:

```bash
py -m streamlit run app.py
```

Run the tests:

```bash
py test_fire_ranking.py
```

## Limits you should know about

1. **Satellites don't watch continuously.** MODIS and VIIRS pass over a given spot only a few times a day (roughly one to four). No hotspot in the data does not mean the fire is out.
2. **Smoke and cloud hide fires.** Thick smoke columns (pyrocumulonimbus, for example) and ordinary cloud cover block the mid-infrared signal, which leads to missed detections.
3. **Canopy hides small fires.** Dense tree cover can mask low-intensity smoldering on the ground.
4. **This is not a dispatch system.** It's meant to help a person decide where to look first, not to make or replace emergency response decisions. Always confirm with ground reports, aerial observation, and the agencies responsible for the area.
