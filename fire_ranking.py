"""
fire_ranking.py - Core analytics, scoring, clustering, and evaluation logic
for NASA FIRMS satellite fire hotspot prioritization.
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


# Standard NASA FIRMS column aliases
COLUMN_ALIASES = {
    "latitude": ["latitude", "lat", "y"],
    "longitude": ["longitude", "lon", "long", "x"],
    "brightness": ["brightness", "bright_ti4", "bright_ti5", "bright_t31", "temp", "temp_k"],
    "confidence": ["confidence", "conf", "confidence_level"],
    "frp": ["frp", "fire_radiative_power", "power"],
    "acq_date": ["acq_date", "date", "acquisition_date"],
    "acq_time": ["acq_time", "time"],
    "satellite": ["satellite", "sat"],
    "daynight": ["daynight", "day_night", "dn"]
}


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names by converting to lower case, trimming,
    and mapping known FIRMS aliases."""
    df = df.copy()
    cleaned_cols = {col: str(col).strip().lower() for col in df.columns}
    df.rename(columns=cleaned_cols, inplace=True)

    mapped_cols = {}
    for standard_name, aliases in COLUMN_ALIASES.items():
        if standard_name in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                mapped_cols[alias] = standard_name
                break

    if mapped_cols:
        df.rename(columns=mapped_cols, inplace=True)

    return df


def sanitize_firms_data(
    df: pd.DataFrame,
    min_confidence: float = 0.0,
    min_frp: float = 0.0,
    daynight_filter: str = "All"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Cleans, parses, and filters raw NASA FIRMS detections."""
    stats = {
        "raw_count": len(df),
        "removed_missing_coords": 0,
        "removed_duplicates": 0,
        "removed_below_confidence": 0,
        "removed_below_frp": 0,
        "final_count": 0,
        "warnings": []
    }

    if df.empty:
        return pd.DataFrame(), stats

    df = normalize_column_names(df)

    # Validate essential coordinates
    if "latitude" not in df.columns or "longitude" not in df.columns:
        raise ValueError("Missing required spatial columns ('latitude' and 'longitude').")

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Drop invalid coordinates (out of range or NaN)
    valid_coords = (
        df["latitude"].notna()
        & df["longitude"].notna()
        & df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
    )
    stats["removed_missing_coords"] = int((~valid_coords).sum())
    df = df[valid_coords].copy()

    # Normalize Brightness (Kelvin)
    if "brightness" in df.columns:
        df["brightness"] = pd.to_numeric(df["brightness"], errors="coerce").fillna(310.0)
    else:
        stats["warnings"].append("Brightness column missing; defaulted to 310.0 K baseline.")
        df["brightness"] = 310.0

    # Normalize FRP (Fire Radiative Power in MW)
    if "frp" in df.columns:
        df["frp"] = pd.to_numeric(df["frp"], errors="coerce").fillna(0.0)
        df["frp"] = df["frp"].clip(lower=0.0)
    else:
        stats["warnings"].append("FRP column missing; defaulted to 0.0 MW.")
        df["frp"] = 0.0

    # Normalize Confidence: handles VIIRS categorical ('l', 'n', 'h') and MODIS 0-100
    if "confidence" in df.columns:
        if df["confidence"].dtype == object or isinstance(df["confidence"].iloc[0] if len(df) else None, str):
            # VIIRS string confidence mapping
            cat_map = {"l": 30.0, "low": 30.0, "n": 65.0, "nominal": 65.0, "h": 90.0, "high": 90.0}
            numeric_conf = df["confidence"].astype(str).str.strip().str.lower().map(cat_map)
            # If not in cat_map, try numeric parse
            fallback_numeric = pd.to_numeric(df["confidence"], errors="coerce")
            df["confidence"] = numeric_conf.fillna(fallback_numeric).fillna(50.0)
        else:
            df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(50.0)
        df["confidence"] = df["confidence"].clip(lower=0.0, upper=100.0)
    else:
        stats["warnings"].append("Confidence column missing; defaulted to nominal 50%.")
        df["confidence"] = 50.0

    # Remove duplicates
    dup_subset = ["latitude", "longitude"]
    if "acq_date" in df.columns and "acq_time" in df.columns:
        dup_subset.extend(["acq_date", "acq_time"])
    initial_len = len(df)
    df = df.drop_duplicates(subset=dup_subset).copy()
    stats["removed_duplicates"] = initial_len - len(df)

    # Filter by confidence
    before_conf = len(df)
    df = df[df["confidence"] >= min_confidence]
    stats["removed_below_confidence"] = before_conf - len(df)

    # Filter by minimum FRP
    before_frp = len(df)
    df = df[df["frp"] >= min_frp]
    stats["removed_below_frp"] = before_frp - len(df)

    # Filter by Day / Night if requested
    if daynight_filter in ("Day", "Night") and "daynight" in df.columns:
        df["daynight"] = df["daynight"].astype(str).str.strip().str.upper()
        target_char = "D" if daynight_filter == "Day" else "N"
        df = df[df["daynight"] == target_char]

    stats["final_count"] = len(df)
    return df.reset_index(drop=True), stats


def assign_spatial_clusters(
    df: pd.DataFrame,
    grid_resolution_deg: float = 0.05
) -> pd.DataFrame:
    """Groups hotspot detections into spatial clusters.
    
    Using grid-binning (e.g. 0.05° is roughly ~5 km at equator).
    Computes cluster detection frequency (repeats), aggregate FRP,
    and cluster spatial bounds.
    """
    if df.empty:
        return df

    df = df.copy()
    decimals = max(1, int(round(-math.log10(grid_resolution_deg)))) if grid_resolution_deg < 1 else 0
    df["lat_bin"] = (df["latitude"] / grid_resolution_deg).round() * grid_resolution_deg
    df["lon_bin"] = (df["longitude"] / grid_resolution_deg).round() * grid_resolution_deg
    df["lat_bin"] = df["lat_bin"].round(decimals)
    df["lon_bin"] = df["lon_bin"].round(decimals)

    cluster_id_series = df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)
    df["cluster_id"] = cluster_id_series

    # Aggregate statistics per cluster
    agg_df = df.groupby("cluster_id").agg(
        cluster_repeats=("latitude", "count"),
        cluster_sum_frp=("frp", "sum"),
        cluster_max_frp=("frp", "max"),
        cluster_max_brightness=("brightness", "max"),
        cluster_mean_conf=("confidence", "mean"),
        centroid_lat=("latitude", "mean"),
        centroid_lon=("longitude", "mean")
    ).reset_index()

    df = df.merge(agg_df, on="cluster_id", how="left")
    return df


def calculate_calibrated_priority(
    df: pd.DataFrame,
    weight_frp: float = 0.35,
    weight_persistence: float = 0.30,
    weight_temp: float = 0.20,
    weight_conf: float = 0.15
) -> pd.DataFrame:
    """Calculates an explainable, normalized 0-100 Multi-Criteria Priority Score.
    
    Components:
    1. Energy / FRP subscore: Log-scaled fire radiative power (MW), reflecting
       exponential fuel consumption rate (1 to 250+ MW).
    2. Persistence subscore: Number of recurrent overpass detections within cluster (1 to 6+).
    3. Thermal Excess subscore: Brightness temperature above ambient baseline (300 K to 400+ K).
    4. Reliability subscore: Detection confidence percentage (0 to 100%).
    """
    if df.empty:
        return df

    df = df.copy()

    # 1. FRP Subscore (Log-scaled; standard in wildfire science for wide dynamic ranges)
    # 0 MW -> 0, 10 MW -> ~43, 50 MW -> ~71, 150 MW -> ~91, 250+ MW -> 100
    ref_frp = 250.0
    log_frp = np.log1p(df["frp"])
    log_max = np.log1p(ref_frp)
    frp_subscore = (log_frp / log_max * 100.0).clip(0.0, 100.0)

    # 2. Persistence / Repeat Subscore
    # 1 detection -> 15 (initial single spot)
    # 2 -> 35, 3 -> 55, 4 -> 75, 5+ -> 95-100
    repeats = df.get("cluster_repeats", pd.Series(1, index=df.index))
    persist_subscore = ((repeats - 1) * 20.0 + 15.0).clip(15.0, 100.0)

    # 3. Thermal Excess Subscore (Kelvin above 300K ambient)
    # Baseline ~300 K (no excess). 350 K -> 50, 400+ K -> 100
    excess_temp = (df["brightness"] - 300.0).clip(lower=0.0)
    temp_subscore = (excess_temp / 1.0).clip(0.0, 100.0)

    # 4. Confidence Subscore (0 to 100%)
    conf_subscore = df["confidence"].clip(0.0, 100.0)

    # Store individual subscores for full explainability
    df["subscore_energy"] = frp_subscore.round(1)
    df["subscore_persistence"] = persist_subscore.round(1)
    df["subscore_thermal"] = temp_subscore.round(1)
    df["subscore_confidence"] = conf_subscore.round(1)

    # Composite normalized score
    total_w = weight_frp + weight_persistence + weight_temp + weight_conf
    if total_w <= 0:
        total_w = 1.0

    w_frp_norm = weight_frp / total_w
    w_persist_norm = weight_persistence / total_w
    w_temp_norm = weight_temp / total_w
    w_conf_norm = weight_conf / total_w

    priority = (
        w_frp_norm * frp_subscore
        + w_persist_norm * persist_subscore
        + w_temp_norm * temp_subscore
        + w_conf_norm * conf_subscore
    ).clip(0.0, 100.0)

    df["priority_score"] = priority.round(1)

    # Assign Priority Tier
    def assign_tier(score: float) -> str:
        if score >= 75.0:
            return "Critical"
        elif score >= 50.0:
            return "High"
        elif score >= 30.0:
            return "Moderate"
        else:
            return "Low / Monitoring"

    df["priority_tier"] = df["priority_score"].apply(assign_tier)

    # Plain-English explainable rationale generator
    def build_rationale(row) -> str:
        tier = row["priority_tier"]
        frp = row["frp"]
        repeats_val = int(row.get("cluster_repeats", 1))
        conf_val = int(row["confidence"])
        b_val = round(row["brightness"], 1)

        notes = []
        if frp >= 100:
            notes.append(f"intense thermal radiative power ({frp:.1f} MW)")
        elif frp >= 35:
            notes.append(f"moderate heat emission ({frp:.1f} MW)")
        else:
            notes.append(f"low radiative energy ({frp:.1f} MW)")

        if repeats_val >= 4:
            notes.append(f"high persistence across {repeats_val} satellite passes")
        elif repeats_val >= 2:
            notes.append(f"recurrent activity ({repeats_val} passes)")
        else:
            notes.append("single isolated detection")

        if conf_val >= 85:
            notes.append(f"high algorithm confidence ({conf_val}%)")
        else:
            notes.append(f"moderate confidence ({conf_val}%)")

        if repeats_val >= 8 and frp < 40:
            notes.append("⚠️ Note: Highly stationary multi-day heat signature; verify if industrial flare/stack.")

        rationale = f"[{tier}] " + "; ".join(notes) + f". (Temp: {b_val}K)"
        return rationale

    df["explainable_reason"] = df.apply(build_rationale, axis=1)
    return df


def calculate_legacy_uncalibrated_priority(df: pd.DataFrame) -> pd.DataFrame:
    """Computes the original uncalibrated priority score for benchmark comparison:
    priority = 0.35 * brightness + 0.25 * confidence + 0.25 * frp + 15 * repeat_count
    """
    df = df.copy()
    repeats = df.get("cluster_repeats", pd.Series(1, index=df.index))
    df["legacy_priority"] = (
        0.35 * df["brightness"]
        + 0.25 * df["confidence"]
        + 0.25 * df["frp"]
        + 15.0 * repeats
    ).round(2)
    return df


def get_top_prioritized_locations(
    df: pd.DataFrame,
    top_n: int = 20,
    deduplicate_by_cluster: bool = True
) -> pd.DataFrame:
    """Returns top ranked locations, optionally picking the peak detection per spatial cluster."""
    if df.empty:
        return df

    sorted_df = df.sort_values("priority_score", ascending=False)
    if deduplicate_by_cluster and "cluster_id" in sorted_df.columns:
        result = sorted_df.drop_duplicates(subset=["cluster_id"])
    else:
        result = sorted_df

    return result.head(top_n).reset_index(drop=True)


def generate_synthetic_evaluation_benchmarks() -> pd.DataFrame:
    """Generates a labelled reference validation dataset containing 5 representative
    ground-truth hotspot archetypes for functional & quantitative algorithm evaluation.
    """
    benchmarks = [
        {
            "case_id": "TC-01",
            "archetype": "Major Wildfire Conflagration",
            "description": "Intense active crown fire with heavy fuel load and continuous multi-pass persistence.",
            "latitude": 39.812,
            "longitude": -121.437,
            "brightness": 412.5,
            "confidence": 98.0,
            "frp": 320.0,
            "cluster_repeats": 7,
            "expected_tier": "Critical",
            "ground_truth_urgency": 100
        },
        {
            "case_id": "TC-02",
            "archetype": "Rapidly Growing Spot Fire",
            "description": "Emerging wildfire flank with significant radiative output detected on recent passes.",
            "latitude": 39.845,
            "longitude": -121.410,
            "brightness": 358.0,
            "confidence": 85.0,
            "frp": 85.0,
            "cluster_repeats": 3,
            "expected_tier": "High",
            "ground_truth_urgency": 75
        },
        {
            "case_id": "TC-03",
            "archetype": "Persistent Industrial Gas Flare / Refinery",
            "description": "Stationary thermal source with high repeat count but localized, moderate radiative energy.",
            "latitude": 29.870,
            "longitude": -95.120,
            "brightness": 345.0,
            "confidence": 90.0,
            "frp": 38.0,
            "cluster_repeats": 14,
            "expected_tier": "Moderate",
            "ground_truth_urgency": 45
        },
        {
            "case_id": "TC-04",
            "archetype": "Controlled Agricultural / Prescribed Burn",
            "description": "Low-intensity planned understory burn detected on a single clear overpass.",
            "latitude": 36.450,
            "longitude": -119.550,
            "brightness": 322.0,
            "confidence": 70.0,
            "frp": 14.5,
            "cluster_repeats": 1,
            "expected_tier": "Low / Monitoring",
            "ground_truth_urgency": 25
        },
        {
            "case_id": "TC-05",
            "archetype": "Solar Reflection Glint / Warm Bare Soil False Alarm",
            "description": "Marginal thermal anomaly near cloud edge with low confidence and near-ambient brightness.",
            "latitude": 34.110,
            "longitude": -115.820,
            "brightness": 308.2,
            "confidence": 32.0,
            "frp": 2.1,
            "cluster_repeats": 1,
            "expected_tier": "Low / Monitoring",
            "ground_truth_urgency": 10
        }
    ]
    return pd.DataFrame(benchmarks)


def run_algorithm_evaluation(
    benchmark_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """Runs a structured functional evaluation comparing:
    1. Calibrated Normalized Multi-Criteria Model
    2. Legacy Uncalibrated Raw Formula
    
    Checks for:
    - Ground-truth ranking alignment (Spearman Rank Correlation)
    - Invariant score bounds [0, 100]
    - Critical Archetype Identification Accuracy
    """
    if benchmark_df is None:
        benchmark_df = generate_synthetic_evaluation_benchmarks()

    df = benchmark_df.copy()
    df = calculate_calibrated_priority(df)
    df = calculate_legacy_uncalibrated_priority(df)

    # Invariant Verification
    all_scores_bounded = bool((df["priority_score"] >= 0.0).all() and (df["priority_score"] <= 100.0).all())
    no_nan_scores = bool(df["priority_score"].notna().all())

    # Tier match check
    tier_matches = (df["priority_tier"] == df["expected_tier"]).sum()
    tier_accuracy = float(tier_matches / len(df))

    # Spearman rank correlation with ground-truth urgency (Pearson correlation on ranks)
    rank_calibrated = df["priority_score"].rank(ascending=False)
    rank_legacy = df["legacy_priority"].rank(ascending=False)
    rank_ground_truth = df["ground_truth_urgency"].rank(ascending=False)

    corr_calibrated = float(rank_calibrated.corr(rank_ground_truth))
    corr_legacy = float(rank_legacy.corr(rank_ground_truth))

    results = {
        "benchmark_count": len(df),
        "all_scores_bounded": all_scores_bounded,
        "no_nan_scores": no_nan_scores,
        "tier_accuracy": tier_accuracy,
        "spearman_correlation_calibrated": round(corr_calibrated, 3),
        "spearman_correlation_legacy": round(corr_legacy, 3),
        "evaluated_dataframe": df
    }
    return results
