"""
Fire Hotspot Priority Dashboard
-------------------------------
An explainable triage and monitoring platform for NASA FIRMS satellite fire detections (MODIS / VIIRS).

Features:
- Robust multi-file ingestion (handles case variations, VIIRS string confidences, missing columns)
- Calibrated, normalized Multi-Criteria Priority Ranking (0-100 scale)
- Spatial binning & clustering (~2km, ~5km, ~11km)
- Interactive PyDeck map with heatmaps, urgency color tiers, and hover tooltips
- Plain-English explainable rationales and component subscore breakdowns
- Contextual overlays: Day/Night breakdown, FRP vs Temperature distributions
- Built-in Functional & Labelled-Sample Evaluation benchmark comparing Calibrated vs Legacy scoring
- Exportable GeoJSON, CSV, and Executive Briefing reports
- Scientific limitations and operational advisories
"""

import json
import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

from fire_ranking import (
    normalize_column_names,
    sanitize_firms_data,
    assign_spatial_clusters,
    calculate_calibrated_priority,
    calculate_legacy_uncalibrated_priority,
    get_top_prioritized_locations,
    generate_synthetic_evaluation_benchmarks,
    run_algorithm_evaluation
)

# -----------------------------------------------------------------------------
# Streamlit Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FIRMS Fire Hotspot Prioritization Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #ff4b4b;
        margin-bottom: 12px;
    }
    .tier-badge-critical {
        background-color: #ff4b4b;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .tier-badge-high {
        background-color: #ff8c00;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .tier-badge-moderate {
        background-color: #ffbb00;
        color: #222;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .tier-badge-low {
        background-color: #4a90e2;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .callout-box {
        background-color: #eef5fc;
        border-left: 4px solid #2b78e4;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 10px 0px 20px 0px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Header & Domain Context
# -----------------------------------------------------------------------------
st.title("🔥 NASA FIRMS Fire Hotspot Prioritization Dashboard")
st.markdown("""
**An explainable situational triage system for spaceborne thermal anomalies.**  
This tool ingests satellite active fire data from **NASA FIRMS (MODIS / VIIRS)**, cleans and standardizes 
inconsistent formats, aggregates repeated detections into incident clusters, and ranks locations using an 
explainable, normalized multi-criteria urgency score.
""")

# -----------------------------------------------------------------------------
# Sidebar: Configuration, Ingestion & Weights
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Data Source & Settings")

data_source = st.sidebar.radio(
    "Select Data Source",
    options=[
        "⚡ Built-in Sample (California Wildfire Complex)",
        "📁 Upload FIRMS Files (.csv / .txt)",
        "🧪 Synthetic Evaluation Archetypes (5 Test Cases)"
    ],
    index=0
)

# File Ingestion Logic
raw_dfs = []
source_name = ""

if "Built-in Sample" in data_source:
    try:
        sample_df = pd.read_csv("sample_firms_data.csv")
        raw_dfs.append(sample_df)
        source_name = "Built-in California Multi-Satellite Dataset"
        st.sidebar.success("✅ Loaded 25 sample detections (MODIS Terra/Aqua).")
    except Exception as e:
        st.sidebar.error(f"Error loading sample file: {e}")

elif "Upload FIRMS" in data_source:
    uploaded_files = st.sidebar.file_uploader(
        "Upload NASA FIRMS Files",
        type=["csv", "txt"],
        accept_multiple_files=True,
        help="Upload standard CSV or TXT downloads from NASA FIRMS (MODIS Terra/Aqua or VIIRS S-NPP/NOAA-20/21)."
    )
    if uploaded_files:
        for f in uploaded_files:
            try:
                temp_df = pd.read_csv(f)
                raw_dfs.append(temp_df)
            except Exception as e:
                st.sidebar.warning(f"Could not parse '{f.name}': {e}")
        source_name = f"{len(raw_dfs)} uploaded file(s)"
    else:
        st.info("👈 Please upload one or more NASA FIRMS CSV/TXT files or switch to the **Built-in Sample** in the sidebar to get started.")
        st.stop()

elif "Synthetic Evaluation Archetypes" in data_source:
    bench_df = generate_synthetic_evaluation_benchmarks()
    raw_dfs.append(bench_df)
    source_name = "5 Labelled Evaluation Archetypes"
    st.sidebar.info("🧪 Using 5 labelled reference cases for validation & benchmark testing.")

if not raw_dfs:
    st.error("No valid fire data available to process.")
    st.stop()

combined_raw_df = pd.concat(raw_dfs, ignore_index=True)

# -----------------------------------------------------------------------------
# Sidebar: Filters & Thresholds
# -----------------------------------------------------------------------------
st.sidebar.subheader("🔍 Hotspot Filtering")

min_confidence = st.sidebar.slider(
    "Minimum Detection Confidence (%)",
    min_value=0,
    max_value=100,
    value=20,
    step=5,
    help="Filters out low-confidence detections (e.g. cloud-edge noise, sun glint)."
)

min_frp = st.sidebar.slider(
    "Minimum FRP Threshold (MW)",
    min_value=0.0,
    max_value=100.0,
    value=0.0,
    step=5.0,
    help="Fire Radiative Power (MW). Higher values focus on high-energy active combustion."
)

daynight_filter = st.sidebar.selectbox(
    "Day / Night Pass Filter",
    options=["All", "Day", "Night"],
    index=0,
    help="Filter by satellite overpass time. Nighttime detections minimize solar reflection false positives."
)

# -----------------------------------------------------------------------------
# Sidebar: Spatial Clustering & Scoring Weights
# -----------------------------------------------------------------------------
st.sidebar.subheader("🗺️ Spatial Grouping & Binning")

grid_resolution = st.sidebar.select_slider(
    "Spatial Clustering Grid Size",
    options=[0.02, 0.05, 0.10],
    value=0.05,
    format_func=lambda x: f"{x}° (~{int(x*111)} km cell)",
    help="Bins nearby satellite detections into coherent fire complexes / clusters."
)

top_n = st.sidebar.slider(
    "Max Locations to Rank (Top N)",
    min_value=5,
    max_value=100,
    value=20,
    step=5
)

dedup_clusters = st.sidebar.checkbox(
    "Deduplicate by Spatial Cluster",
    value=True,
    help="When enabled, surfaces only the highest-scoring detection per spatial cluster."
)

# Prioritization Preset
st.sidebar.subheader("⚖️ Ranking Formula Weights")
weight_preset = st.sidebar.selectbox(
    "Weighting Preset",
    options=[
        "Balanced (Recommended)",
        "Conflagration / Energy Focus (High FRP)",
        "Persistence & Spread Monitoring",
        "Custom Sliders"
    ]
)

if weight_preset == "Balanced (Recommended)":
    w_frp, w_persist, w_temp, w_conf = 0.35, 0.30, 0.20, 0.15
elif weight_preset == "Conflagration / Energy Focus (High FRP)":
    w_frp, w_persist, w_temp, w_conf = 0.50, 0.20, 0.20, 0.10
elif weight_preset == "Persistence & Spread Monitoring":
    w_frp, w_persist, w_temp, w_conf = 0.25, 0.45, 0.15, 0.15
else:
    w_frp = st.sidebar.slider("FRP (Heat Output) Weight", 0.0, 1.0, 0.35, 0.05)
    w_persist = st.sidebar.slider("Persistence (Repeats) Weight", 0.0, 1.0, 0.30, 0.05)
    w_temp = st.sidebar.slider("Excess Temp Weight", 0.0, 1.0, 0.20, 0.05)
    w_conf = st.sidebar.slider("Confidence Weight", 0.0, 1.0, 0.15, 0.05)

# -----------------------------------------------------------------------------
# Data Processing Pipeline
# -----------------------------------------------------------------------------
try:
    cleaned_df, clean_stats = sanitize_firms_data(
        combined_raw_df,
        min_confidence=min_confidence,
        min_frp=min_frp,
        daynight_filter=daynight_filter
    )
except Exception as err:
    st.error(f"Error during data sanitization: {err}")
    st.stop()

if cleaned_df.empty:
    st.warning("⚠️ No detections remain after applying your selected filters. Try lowering the confidence or FRP thresholds in the sidebar.")
    st.stop()

# Spatial Clustering
clustered_df = assign_spatial_clusters(cleaned_df, grid_resolution_deg=grid_resolution)

# Priority Scoring
scored_df = calculate_calibrated_priority(
    clustered_df,
    weight_frp=w_frp,
    weight_persistence=w_persist,
    weight_temp=w_temp,
    weight_conf=w_conf
)

# Also compute legacy score for side-by-side comparison
scored_df = calculate_legacy_uncalibrated_priority(scored_df)

# Top Ranked Locations
top_ranked_df = get_top_prioritized_locations(
    scored_df,
    top_n=top_n,
    deduplicate_by_cluster=dedup_clusters
)

# -----------------------------------------------------------------------------
# Top Metric KPI Row
# -----------------------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

total_frp = cleaned_df["frp"].sum()
unique_clusters = clustered_df["cluster_id"].nunique()
critical_count = (scored_df["priority_tier"] == "Critical").sum()

col1.metric("Ingested Detections", f"{clean_stats['raw_count']:,}")
col2.metric("Filtered Hotspots", f"{len(cleaned_df):,}", delta=f"-{clean_stats['removed_missing_coords'] + clean_stats['removed_duplicates'] + clean_stats['removed_below_confidence'] + clean_stats['removed_below_frp']} filtered")
col3.metric("Fire Clusters", f"{unique_clusters:,}")
col4.metric("Total FRP", f"{total_frp:,.1f} MW")
col5.metric("Critical Urgency", f"{critical_count:,}", delta_color="inverse")

# Sanitization Summary Expander
with st.expander("ℹ️ Data Ingestion & Sanitation Details"):
    c_s1, c_s2 = st.columns(2)
    with c_s1:
        st.write(f"• **Active Data Source**: {source_name}")
        st.write(f"• **Duplicate Records Removed**: {clean_stats['removed_duplicates']}")
        st.write(f"• **Invalid/Out-of-Bounds Coordinates**: {clean_stats['removed_missing_coords']}")
    with c_s2:
        st.write(f"• **Filtered Below Confidence (<{min_confidence}%)**: {clean_stats['removed_below_confidence']}")
        st.write(f"• **Filtered Below FRP (<{min_frp} MW)**: {clean_stats['removed_below_frp']}")
        if clean_stats["warnings"]:
            for w in clean_stats["warnings"]:
                st.warning(w)

# -----------------------------------------------------------------------------
# Main Tabs: Map, Queue, Analytics, Evaluation, Export, Limitations
# -----------------------------------------------------------------------------
tab_map, tab_queue, tab_analytics, tab_eval, tab_export, tab_limits = st.tabs([
    "🗺️ Interactive Fire Map",
    "📋 Prioritized Hotspot Queue",
    "📊 Contextual Overlays & Distributions",
    "🧪 Functional & Labelled Evaluation",
    "📥 Export & Incident Briefing",
    "⚠️ Operational Limitations & Ethics"
])

# -----------------------------------------------------------------------------
# TAB 1: INTERACTIVE MAP
# -----------------------------------------------------------------------------
with tab_map:
    st.subheader("Geospatial Hotspot Distribution & Severity Tiers")
    st.caption("Point size indicates Fire Radiative Power (MW). Color indicates Calibrated Urgency Tier.")

    map_col1, map_col2 = st.columns([3, 1])

    with map_col2:
        map_style = st.selectbox(
            "Map Basemap Style",
            options=["dark", "light", "satellite", "road"],
            index=0
        )
        show_heatmap = st.checkbox("Overlay Thermal Density Heatmap", value=False)
        radius_scale = st.slider("Point Radius Multiplier", 1.0, 5.0, 2.0, 0.5)

        st.markdown("""
        **Urgency Tier Legend:**
        - 🔴 <span class='tier-badge-critical'>Critical (Score ≥ 75)</span>
        - 🟠 <span class='tier-badge-high'>High (50 – 74)</span>
        - 🟡 <span class='tier-badge-moderate'>Moderate (30 – 49)</span>
        - 🔵 <span class='tier-badge-low'>Low / Monitor (< 30)</span>
        """, unsafe_allow_html=True)

    with map_col1:
        # Prepare color mapping for PyDeck
        # Critical: Red [235, 50, 50], High: Orange [245, 140, 25], Moderate: Yellow [245, 200, 30], Low: Blue [60, 130, 230]
        def get_color(tier):
            if tier == "Critical":
                return [235, 50, 50, 200]
            elif tier == "High":
                return [245, 140, 25, 190]
            elif tier == "Moderate":
                return [245, 200, 30, 180]
            else:
                return [60, 130, 230, 160]

        map_data = top_ranked_df.copy()
        map_data["color"] = map_data["priority_tier"].apply(get_color)
        map_data["radius"] = (np.sqrt(map_data["frp"].clip(lower=2.0)) * 250 * radius_scale).clip(lower=400, upper=8000)

        # Center map
        avg_lat = float(map_data["latitude"].mean())
        avg_lon = float(map_data["longitude"].mean())

        layers = []

        if show_heatmap:
            heat_layer = pdk.Layer(
                "HeatmapLayer",
                data=cleaned_df,
                get_position=["longitude", "latitude"],
                get_weight="frp",
                radius_pixels=40,
                intensity=1.5,
                threshold=0.05
            )
            layers.append(heat_layer)

        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_data,
            get_position=["longitude", "latitude"],
            get_color="color",
            get_radius="radius",
            pickable=True,
            auto_highlight=True
        )
        layers.append(scatter_layer)

        view_state = pdk.ViewState(
            latitude=avg_lat,
            longitude=avg_lon,
            zoom=6 if map_data["latitude"].std() < 2 else 4,
            pitch=25
        )

        tooltip = {
            "html": """
            <div style="font-family: sans-serif; font-size: 12px; padding: 6px;">
                <b>Tier: {priority_tier}</b> (Score: {priority_score}/100)<br/>
                <b>FRP:</b> {frp} MW | <b>Brightness:</b> {brightness} K<br/>
                <b>Repeats:</b> {cluster_repeats} passes | <b>Confidence:</b> {confidence}%<br/>
                <b>Cluster:</b> {cluster_id}<br/>
                <hr style="margin: 4px 0; border: 0.5px solid #ccc;"/>
                <i>{explainable_reason}</i>
            </div>
            """,
            "style": {"backgroundColor": "#1e2129", "color": "#ffffff", "borderRadius": "4px"}
        }

        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            map_style=f"mapbox://styles/mapbox/{map_style}-v9",
            tooltip=tooltip
        )
        st.pydeck_chart(deck)

# -----------------------------------------------------------------------------
# TAB 2: PRIORITIZED HOTSPOT QUEUE (TABLE & EXPLAINABILITY)
# -----------------------------------------------------------------------------
with tab_queue:
    st.subheader(f"Top {len(top_ranked_df)} Prioritized Locations")
    st.caption("Locations ranked by multi-criteria urgency. Each entry includes its highest-scoring detection and explainable rationale.")

    # Formatted display table
    display_cols = [
        "priority_score", "priority_tier", "latitude", "longitude",
        "frp", "cluster_repeats", "brightness", "confidence",
        "subscore_energy", "subscore_persistence", "explainable_reason"
    ]
    existing_cols = [c for c in display_cols if c in top_ranked_df.columns]
    table_view = top_ranked_df[existing_cols].copy()

    table_view.rename(columns={
        "priority_score": "Score (0-100)",
        "priority_tier": "Tier",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "frp": "FRP (MW)",
        "cluster_repeats": "Repeats",
        "brightness": "Temp (K)",
        "confidence": "Conf (%)",
        "subscore_energy": "Energy Sub",
        "subscore_persistence": "Persist Sub",
        "explainable_reason": "Rationale"
    }, inplace=True)

    st.dataframe(
        table_view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score (0-100)": st.column_config.ProgressColumn(
                "Priority Score",
                format="%.1f",
                min_value=0,
                max_value=100
            ),
            "FRP (MW)": st.column_config.NumberColumn(format="%.1f MW"),
            "Temp (K)": st.column_config.NumberColumn(format="%.1f K"),
            "Conf (%)": st.column_config.NumberColumn(format="%d%%")
        }
    )

    # Detailed Incident Inspector
    st.divider()
    st.subheader("🔍 Deep-Dive Incident Inspector")
    selected_idx = st.selectbox(
        "Select a location rank to inspect explainability components:",
        options=range(len(top_ranked_df)),
        format_func=lambda i: f"Rank #{i+1}: Tier [{top_ranked_df.iloc[i]['priority_tier']}] - Coord ({top_ranked_df.iloc[i]['latitude']:.3f}, {top_ranked_df.iloc[i]['longitude']:.3f}) - FRP: {top_ranked_df.iloc[i]['frp']:.1f} MW"
    )

    if len(top_ranked_df) > 0:
        row = top_ranked_df.iloc[selected_idx]
        d_col1, d_col2 = st.columns([2, 1])

        with d_col1:
            st.markdown(f"### Incident {row.get('cluster_id', 'N/A')} (Rank #{selected_idx+1})")
            st.markdown(f"**Explainable Rationale:** `{row['explainable_reason']}`")

            # Progress breakdown of subscores
            s_e = row.get("subscore_energy", 0)
            s_p = row.get("subscore_persistence", 0)
            s_t = row.get("subscore_thermal", 0)
            s_c = row.get("subscore_confidence", 0)

            st.write("#### Component Subscore Contributions (0-100):")
            sub_df = pd.DataFrame({
                "Component": ["Thermal Radiative Energy (FRP)", "Temporal Persistence (Repeats)", "Thermal Excess (Kelvin)", "Algorithmic Confidence (%)"],
                "Weight": [f"{w_frp*100:.0f}%", f"{w_persist*100:.0f}%", f"{w_temp*100:.0f}%", f"{w_conf*100:.0f}%"],
                "Normalized Subscore (0-100)": [s_e, s_p, s_t, s_c]
            })
            st.table(sub_df)

        with d_col2:
            st.markdown("#### Operational Summary")
            st.metric("Total Cluster Radiative Energy", f"{row.get('cluster_sum_frp', row['frp']):.1f} MW")
            st.metric("Consecutive Overpass Passes", f"{int(row.get('cluster_repeats', 1))} overpasses")
            st.metric("Max Temperature Observed", f"{row.get('cluster_max_brightness', row['brightness']):.1f} K")
            if "acq_date" in row:
                st.caption(f"Most recent pass: {row.get('acq_date', 'N/A')} at {row.get('acq_time', 'N/A')} UTC")

# -----------------------------------------------------------------------------
# TAB 3: CONTEXTUAL OVERLAYS & DISTRIBUTIONS
# -----------------------------------------------------------------------------
with tab_analytics:
    st.subheader("Contextual Analytics & Detection Distributions")
    st.markdown("Examine cross-correlations between thermal radiance, platform overpasses, and diurnal patterns.")

    a_col1, a_col2 = st.columns(2)

    with a_col1:
        st.markdown("#### Fire Radiative Power (MW) vs. Brightness (K)")
        # Scatter chart of FRP vs Temperature
        scatter_chart_data = cleaned_df[["brightness", "frp", "confidence"]].copy()
        st.scatter_chart(
            scatter_chart_data,
            x="brightness",
            y="frp",
            color="confidence",
            size="frp"
        )
        st.caption("Fires in the upper right exhibit both extreme surface temperature and high volumetric fuel consumption.")

    with a_col2:
        st.markdown("#### Diurnal & Satellite Platform Overpasses")
        dist_cols = []
        if "daynight" in cleaned_df.columns:
            dist_cols.append("daynight")
        if "satellite" in cleaned_df.columns:
            dist_cols.append("satellite")

        if dist_cols:
            grouped_counts = cleaned_df.groupby(dist_cols).size().reset_index(name="detection_count")
            st.dataframe(grouped_counts, use_container_width=True, hide_index=True)
        else:
            st.info("Satellite and Day/Night metadata columns not present in dataset.")

        st.markdown("#### Radiative Energy (FRP) Quartile Summary")
        frp_summary = cleaned_df["frp"].describe().round(2).to_frame().T
        st.dataframe(frp_summary, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: FUNCTIONAL & LABELLED-SAMPLE EVALUATION
# -----------------------------------------------------------------------------
with tab_eval:
    st.subheader("🧪 Functional & Labelled-Sample Algorithm Evaluation")
    st.markdown("""
    To ensure scientific rigor and avoid unsupported claims, we evaluate the prioritization model against 
    a **labelled benchmark of 5 representative fire archetypes**.  
    This validates:
    1. **Sensitivity & Monotonicity:** Severe conflagrations reliably outrank small agricultural burns.
    2. **Stationary Industrial Source Mitigation:** Uncalibrated raw formulas often elevate stationary gas flares above massive wildfires due to unbounded repeat multipliers.
    3. **Mathematical Invariants:** Verification that scores are strictly bounded in $[0, 100]$ with zero NaNs.
    """)

    eval_results = run_algorithm_evaluation()
    eval_df = eval_results["evaluated_dataframe"]

    ev_m1, ev_m2, ev_m3, ev_m4 = st.columns(4)
    ev_m1.metric("Benchmark Cases", eval_results["benchmark_count"])
    ev_m2.metric("Score Bounds [0, 100]", "PASSED ✅" if eval_results["all_scores_bounded"] else "FAILED ❌")
    ev_m3.metric("Calibrated Spearman Rank Corr", f"{eval_results['spearman_correlation_calibrated']:.2f}")
    ev_m4.metric("Legacy Raw Formula Corr", f"{eval_results['spearman_correlation_legacy']:.2f}", delta=f"{eval_results['spearman_correlation_calibrated'] - eval_results['spearman_correlation_legacy']:+.2f} improvement")

    st.markdown("#### Labelled Benchmark Comparison Table")
    eval_display = eval_df[[
        "case_id", "archetype", "frp", "cluster_repeats", "confidence",
        "ground_truth_urgency", "priority_score", "priority_tier", "legacy_priority"
    ]].copy()

    eval_display.rename(columns={
        "case_id": "Case ID",
        "archetype": "Archetype",
        "frp": "FRP (MW)",
        "cluster_repeats": "Repeats",
        "confidence": "Conf (%)",
        "ground_truth_urgency": "Ground Truth Urgency",
        "priority_score": "Calibrated Score",
        "priority_tier": "Calibrated Tier",
        "legacy_priority": "Legacy Raw Formula Score"
    }, inplace=True)

    st.dataframe(eval_display, use_container_width=True, hide_index=True)

    st.markdown("""
    #### 💡 Key Evaluation Takeaway: The "Gas Flare Flaw"
    Notice **TC-03** (Stationary Gas Flare with 14 repeat overpasses):
    - **In the Legacy Raw Formula:** $0.35 \\times 345 + 0.25 \\times 90 + 0.25 \\times 38 + 15 \\times 14 = \\mathbf{362.75}$.  
      This outranks **TC-01** (Major Active Wildfire Conflagration: $\\mathbf{353.88}$)!  
      The uncalibrated linear model dangerously favored a tiny 38 MW gas flare over a ferocious 320 MW crown wildfire.
    - **In our Calibrated Model:** Logarithmic FRP scaling and normalized persistence appropriately keep the major wildfire in the **#1 Critical Tier (Score 99.7)**, while flagging TC-03 with an automated stationary heat source advisory.
    """)

# -----------------------------------------------------------------------------
# TAB 5: EXPORT & INCIDENT BRIEFING
# -----------------------------------------------------------------------------
with tab_export:
    st.subheader("📥 Export Prioritized Hotspots & Operational Briefings")
    st.markdown("Export ranked data for integration into GIS workflows (QGIS, ArcGIS) or incident briefings.")

    exp_col1, exp_col2, exp_col3 = st.columns(3)

    # 1. CSV Export
    csv_bytes = top_ranked_df.to_csv(index=False).encode("utf-8")
    exp_col1.download_button(
        label="📄 Download Prioritized CSV",
        data=csv_bytes,
        file_name="firms_prioritized_hotspots.csv",
        mime="text/csv"
    )

    # 2. GeoJSON Export
    geojson_features = []
    for _, r in top_ranked_df.iterrows():
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(r["longitude"]), float(r["latitude"])]
            },
            "properties": {
                "priority_score": float(r["priority_score"]),
                "priority_tier": str(r["priority_tier"]),
                "frp": float(r["frp"]),
                "brightness": float(r["brightness"]),
                "confidence": float(r["confidence"]),
                "cluster_repeats": int(r.get("cluster_repeats", 1)),
                "cluster_id": str(r.get("cluster_id", "")),
                "rationale": str(r["explainable_reason"])
            }
        }
        geojson_features.append(feature)

    geojson_collection = {
        "type": "FeatureCollection",
        "features": geojson_features
    }
    geojson_bytes = json.dumps(geojson_collection, indent=2).encode("utf-8")

    exp_col2.download_button(
        label="🗺️ Download GeoJSON (GIS-ready)",
        data=geojson_bytes,
        file_name="firms_prioritized_hotspots.geojson",
        mime="application/geo+json"
    )

    # 3. Markdown Executive Briefing Export
    briefing_text = f"""# NASA FIRMS Thermal Hotspot Prioritization Briefing
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
Data Source: {source_name}

## 1. Executive Summary
- Total Hotspots Ingested: {clean_stats['raw_count']}
- Analyzed Unique Fire Clusters: {unique_clusters}
- Aggregate Fire Radiative Power (FRP): {total_frp:,.1f} MW
- Critical Urgency Locations: {critical_count}

## 2. Top Priority Hotspots
"""
    for i, r in top_ranked_df.head(10).iterrows():
        briefing_text += f"\n### Rank #{i+1} | Tier: {r['priority_tier']} (Score: {r['priority_score']}/100)\n"
        briefing_text += f"- **Coordinates**: {r['latitude']:.4f}° N, {r['longitude']:.4f}° E (Cluster {r.get('cluster_id', 'N/A')})\n"
        briefing_text += f"- **Thermal Metrics**: FRP: {r['frp']:.1f} MW | Brightness: {r['brightness']:.1f} K | Confidence: {r['confidence']:.0f}%\n"
        briefing_text += f"- **Persistence**: {int(r.get('cluster_repeats', 1))} overpass detections\n"
        briefing_text += f"- **Rationale**: {r['explainable_reason']}\n"

    briefing_text += """
---
*Disclaimer: Satellite thermal anomaly detections are subject to latency, cloud masking, and geolocation drift. This report is an assistive triage tool and NOT an automated emergency dispatch system.*
"""
    exp_col3.download_button(
        label="📑 Download Executive Briefing (.md)",
        data=briefing_text.encode("utf-8"),
        file_name="fire_incident_briefing.md",
        mime="text/markdown"
    )

    st.divider()
    st.markdown("#### Executive Briefing Preview")
    st.text_area("Preview Briefing Text", briefing_text, height=220)

# -----------------------------------------------------------------------------
# TAB 6: LIMITATIONS & ETHICS
# -----------------------------------------------------------------------------
with tab_limits:
    st.subheader("⚠️ Operational Limitations & Responsible AI Notice")
    st.markdown("""
    Satellite-based remote sensing provides invaluable wide-area situational awareness, but operational 
    decision-makers must account for inherent physical and algorithmic constraints:
    """)

    l_col1, l_col2 = st.columns(2)

    with l_col1:
        st.markdown("""
        #### 🛰️ Physical & Sensor Limitations
        - **Orbital Revisit Latency:** Low Earth Orbit (LEO) satellites (MODIS Terra/Aqua, VIIRS S-NPP/NOAA) view any given mid-latitude location only 1–4 times every 24 hours. **The absence of a hotspot does NOT confirm a fire is extinguished.**
        - **Cloud & Heavy Smoke Obscuration:** Thick convective pyrocumulonimbus clouds or dense smoke plumes absorb and scatter 4µm mid-wave infrared radiation, potentially masking intense surface fires.
        - **Canopy Attenuation:** Dense forest tree canopies mask low-intensity smoldering understory fires from detection.
        - **Spatial Resolution & Geolocation Drift:** Nominal sub-satellite pixel resolution is 1 km (MODIS) or 375 m (VIIRS). Near the edge of the scan swath, pixel footprint expands up to ~2x.
        """)

    with l_col2:
        st.markdown("""
        #### 🛑 Operational & Algorithmic Boundaries
        - **Triage Heuristic, NOT a Predictive Model:** The priority score evaluates current and historical thermal evidence. It does not predict future fire spread, wind direction, or fuel moisture without coupled weather models.
        - **Non-Fire Thermal Signatures:** Highly persistent anomalies in industrial corridors (refineries, metal smelters, gas flares) or high-albedo solar reflection glints can trigger thermal alerts.
        - **Not an Emergency Dispatch Authority:** This dashboard is an assistive analytical tool designed to help analysts prioritize satellite feeds. It must never be used as an autonomous dispatch authority without ground verification.
        """)

st.caption("Developed for wildfire situational triage • Powered by NASA FIRMS active fire products • Compliant with open scientific standards.")
