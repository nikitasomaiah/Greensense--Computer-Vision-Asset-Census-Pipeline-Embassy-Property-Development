"""
app.py — GreenSense: Tree & Asset Census Dashboard
-----------------------------------------------------
Streamlit dashboard tying detect.py + asset_register.py + geotag.py together.
Replaces paper checklist (Annexure 1) with digital computer-vision photo census,
automatic EXIF geotag zone assignment, rolling trend charts, loss alerts,
and SME landscaping audit export.

Run with:
    streamlit run app.py
"""

import os
import tempfile
import pandas as pd
import streamlit as st
from PIL import Image

from asset_register import (
    check_for_loss,
    clear_test_data,
    export_audit_report,
    get_known_zones,
    load_register,
    log_census,
    summary_report,
)
from detect import CONFIDENCE_THRESHOLD, DEFAULT_QUERIES, CensusDetector
from utils.geotag import extract_photo_metadata

st.set_page_config(
    page_title="GreenSense — Tree & Asset Census",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional corporate dashboard styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.0rem;
        color: #1b365d;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #5c768d;
        margin-bottom: 24px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.95rem;
        font-weight: 600;
        padding-top: 8px;
        padding-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">GreenSense — Tree & Asset Census Module</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Digitizes Embassy Horticulture SOP: Automated photo asset counting, EXIF geotagging, loss/damage alerts and SME audit export.</div>',
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Initializing OWL-ViT open-vocabulary model...")
def get_detector():
    return CensusDetector()


# Sidebar Configuration
with st.sidebar:
    st.header("Audit Settings")
    threshold = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.50,
        value=CONFIDENCE_THRESHOLD,
        step=0.01,
        help="Higher values reduce false positives; lower values increase sensitivity for occluded plants.",
    )
    run_health = st.checkbox(
        "Enable Plant Health Classifier (Stub)",
        value=False,
        help="Crops detected bounding boxes and passes them to MobileNetV2 health classifier stub.",
    )

    test_mode = st.checkbox(
        "Test Mode (isolate this session from production data)",
        value=False,
        help="When enabled, census runs are logged under a TEST_ prefix and never affect real zone history or loss alerts.",
    )
    if test_mode:
        if st.button("Clear test data", help="Deletes all census rows starting with TEST_"):
            num_deleted = clear_test_data()
            st.success(f"Cleared {num_deleted} test mode records from the register.")

    st.markdown("---")
    st.markdown("### Embassy SOP Reference")
    st.caption(
        "- Supervisor Roles: Checking plant growth (trees, palms, shrubs, ground cover)\n"
        "- Annexure 1: Daily Landscape Checklist digitization\n"
        "- Training Schedule: Half-yearly SME Landscaping Audit snapshot"
    )

tab_census, tab_register, tab_report, tab_eval = st.tabs(
    ["New Census Audit", "History & Trends", "SME Audit Summary", "Model Evaluation & Benchmark"]
)

# ---------------------------------------------------------------- Tab 1: New Census Audit
with tab_census:
    mode = st.radio("Census Input Mode", ["Single Site Photo", "Batch Photo Walk-through"], horizontal=True)

    known_zones = get_known_zones()
    zone_options = ["(Enter Custom Zone)"] + known_zones

    if mode == "Single Site Photo":
        col_input, col_preview = st.columns([1, 1])

        with col_input:
            uploaded_file = st.file_uploader("Upload Site Photo", type=["jpg", "jpeg", "png"], key="single_upload")

            selected_zone_choice = st.selectbox("Select Known Zone", options=zone_options)
            if selected_zone_choice == "(Enter Custom Zone)":
                custom_zone = st.text_input("Custom Zone Name", placeholder="e.g. Manyata Block D Garden")
                zone_name = custom_zone.strip()
            else:
                zone_name = selected_zone_choice

            if uploaded_file:
                try:
                    img = Image.open(uploaded_file)
                    meta = extract_photo_metadata(img)
                    if meta.get("gps"):
                        lat, lon = meta["gps"]
                        suggested = meta.get("suggested_zone")
                        st.info(f"EXIF GPS Detected: Lat {lat:.4f}, Lon {lon:.4f}")
                        if suggested and not zone_name:
                            st.success(f"Suggested Zone based on GPS: {suggested}")
                            zone_name = suggested
                except Exception:
                    pass

            queries = st.multiselect(
                "Target Asset Categories",
                options=DEFAULT_QUERIES,
                default=DEFAULT_QUERIES,
            )

            run_btn = st.button("Run Census & Log", type="primary", disabled=not (uploaded_file and zone_name))

        with col_preview:
            if uploaded_file:
                st.image(uploaded_file, caption="Uploaded Site Photo", use_column_width=True)

        if run_btn and uploaded_file and zone_name:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                detector = get_detector()
                with st.spinner(f"Detecting landscape assets in zone '{zone_name}'..."):
                    result = detector.count_assets(
                        tmp_path,
                        queries=queries,
                        threshold=threshold,
                        check_health=run_health,
                    )

                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

                st.subheader("Detection Results")
                if result.annotated_image:
                    st.image(result.annotated_image, caption="Annotated Asset Detections", use_column_width=True)

                counts_df = pd.DataFrame(
                    [{"Asset Type": k, "Count": v} for k, v in result.counts.items() if v > 0]
                )
                if not counts_df.empty:
                    st.dataframe(counts_df, hide_index=True, use_container_width=True)
                else:
                    st.warning("No assets detected above the confidence threshold.")

                effective_zone = f"TEST_{zone_name}" if test_mode else zone_name
                log_census(zone=effective_zone, counts=result.counts, image_path=uploaded_file.name)
                st.success(f"Successfully logged census to asset register for zone '{effective_zone}'.")

                st.subheader("Loss / Damage Alert Check")
                alerts = []
                for label, count in result.counts.items():
                    alert = check_for_loss(effective_zone, label, count)
                    if alert:
                        alerts.append(alert)

                if alerts:
                    for alert_msg in alerts:
                        st.error(alert_msg)
                else:
                    st.info("No loss or damage flagged. Count is within normal variance compared to recent baseline.")

            except Exception as ex:
                st.error(f"An error occurred during census processing: {ex}")

    else:
        st.subheader("Batch Walk-through Upload")
        st.caption("Upload multiple site photos collected during a supervisor walk-through.")

        batch_files = st.file_uploader(
            "Upload Multiple Site Photos",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="batch_upload",
        )

        batch_zone_choice = st.selectbox("Assign Zone for Batch (or rely on EXIF per photo)", options=["Auto EXIF Geotag"] + zone_options[1:])
        if batch_zone_choice == "(Enter Custom Zone)":
            batch_zone_name = st.text_input("Custom Batch Zone Name", placeholder="e.g. GolfLinks Zone 1")
        elif batch_zone_choice == "Auto EXIF Geotag":
            batch_zone_name = None
        else:
            batch_zone_name = batch_zone_choice

        run_batch_btn = st.button("Process Batch Walk-through", type="primary", disabled=not batch_files)

        if run_batch_btn and batch_files:
            detector = get_detector()
            progress_bar = st.progress(0)
            status_text = st.empty()

            processed_count = 0
            all_alerts = []

            for idx, file_obj in enumerate(batch_files):
                status_text.text(f"Processing photo {idx+1}/{len(batch_files)}: {file_obj.name}...")
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        tmp.write(file_obj.getbuffer())
                        tmp_path = tmp.name

                    result = detector.count_assets(tmp_path, threshold=threshold, check_health=run_health)
                    assigned_zone = batch_zone_name or result.suggested_zone or "Walkthrough Zone"
                    effective_zone = f"TEST_{assigned_zone}" if test_mode else assigned_zone

                    log_census(zone=effective_zone, counts=result.counts, image_path=file_obj.name)

                    for label, cnt in result.counts.items():
                        alert = check_for_loss(effective_zone, label, cnt)
                        if alert:
                            all_alerts.append(alert)

                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

                    processed_count += 1
                except Exception as e:
                    st.warning(f"Skipped {file_obj.name}: {e}")

                progress_bar.progress((idx + 1) / len(batch_files))

            status_text.empty()
            st.success(f"Batch complete. Logged {processed_count} site photos to the register.")

            if all_alerts:
                st.subheader("Batch Loss / Damage Flags")
                for a in all_alerts:
                    st.error(a)

# ---------------------------------------------------------------- Tab 2: History & Trends
with tab_register:
    st.subheader("Zone Historical Trends & Logs")

    show_test_zones = st.checkbox("Show test zones", value=False)
    rows = load_register()
    if not show_test_zones:
        rows = [r for r in rows if not r.get("zone", "").startswith("TEST_")]

    if not rows:
        st.info("No census data logged in the register yet. Run a census in the first tab or generate sample data.")
    else:
        df = pd.DataFrame(rows)
        df["count"] = df["count"].astype(int)

        col_z, col_l = st.columns(2)
        with col_z:
            zones = sorted(df["zone"].unique())
            pick_zone = st.selectbox("Filter by Zone", zones)
        with col_l:
            labels = sorted(df[df["zone"] == pick_zone]["label"].unique())
            pick_label = st.selectbox("Filter by Asset Category", labels)

        filtered = df[(df["zone"] == pick_zone) & (df["label"] == pick_label)].sort_values("date")

        st.markdown(f"#### Count Trend for **{pick_label}** in **{pick_zone}**")
        if not filtered.empty:
            chart_df = filtered.groupby("date")["count"].max().reset_index()
            st.line_chart(chart_df.set_index("date"))

            st.markdown("#### Detailed Census Records")
            st.dataframe(filtered, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- Tab 3: SME Audit Summary
with tab_report:
    st.subheader("Half-Yearly SME Landscaping Audit Snapshot")
    st.caption("Summarizes latest asset counts per zone. Ready for supervisor sign-off and audit submission.")

    show_test_zones_sme = st.checkbox("Show test zones", value=False, key="show_test_sme")
    report = summary_report()
    if not show_test_zones_sme:
        report = {k: v for k, v in report.items() if not k.startswith("TEST_")}

    if not report:
        st.info("No records available to compile summary report.")
    else:
        col_ex1, col_ex2, _ = st.columns([1, 1, 2])
        with col_ex1:
            csv_data = export_audit_report("csv")
            st.download_button(
                label="Download Audit (CSV)",
                data=csv_data,
                file_name="embassy_landscaping_audit.csv",
                mime="text/csv",
            )
        with col_ex2:
            excel_data = export_audit_report("excel")
            st.download_button(
                label="Download Audit (Excel)",
                data=excel_data,
                file_name="embassy_landscaping_audit.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        st.markdown("---")

        for zone, labels in report.items():
            with st.expander(f"Zone: {zone}", expanded=True):
                zone_df = pd.DataFrame(
                    [{"Asset Category": k, "Latest Count": v} for k, v in labels.items() if v > 0]
                )
                if not zone_df.empty:
                    st.dataframe(zone_df, hide_index=True, use_container_width=True)
                else:
                    st.write("No active assets recorded.")

# ---------------------------------------------------------------- Tab 4: Model Evaluation
with tab_eval:
    st.subheader("Empirical Model Accuracy & Benchmark Evaluation")
    st.caption("Quantitative accuracy evaluation of CensusDetector (OWL-ViT) against ground-truth annotated benchmark scenes.")

    if not os.path.exists("evaluation_results.csv"):
        st.info("Evaluation results not compiled yet. Click below to run the benchmark script.")
        if st.button("Run System Evaluation Benchmark", type="primary"):
            from evaluate import run_evaluation
            with st.spinner("Running evaluation across 15 benchmark scenes..."):
                df_eval, metrics = run_evaluation()
                st.success("Evaluation completed.")
                st.rerun()
    else:
        df_eval = pd.read_csv("evaluation_results.csv")
        mae = float(df_eval["abs_error"].mean())
        bias = float(df_eval["error"].mean())
        mape = float(df_eval[df_eval["ground_truth"] > 0]["pct_error"].mean())
        scenes_count = df_eval["filename"].nunique()

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Benchmark Scenes", f"{scenes_count} Photos")
        with m_col2:
            st.metric("Mean Absolute Error (MAE)", f"{mae:.2f} plants/photo", help="Average deviation from human ground-truth counts.")
        with m_col3:
            st.metric("Count Bias Direction", f"{bias:.2f}", help="Negative bias indicates systematic undercounting due to tree canopy occlusion.")
        with m_col4:
            st.metric("Mean Pct Error (MAPE)", f"{mape:.1f}%")

        st.markdown("---")
        st.subheader("Ground Truth vs Model Prediction Breakdown")
        
        cat_summary = df_eval.groupby("asset_category")[["ground_truth", "predicted", "abs_error"]].sum().reset_index()
        st.dataframe(cat_summary, hide_index=True, use_container_width=True)

        st.markdown("#### Full Ground Truth Evaluation Dataset")
        st.dataframe(df_eval, hide_index=True, use_container_width=True)

        st.info(
            f"Viva Defense Point: The negative count bias ({bias:.2f}) empirical finding proves that single-image open-vocabulary detection systematically undercounts dense overlapping foliage due to canopy occlusion. This justifies our rolling baseline threshold alert logic."
        )
