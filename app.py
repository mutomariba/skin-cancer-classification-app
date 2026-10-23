import io
from pathlib import Path
import html
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
API_URL = "https://skin-cancer-classification-app-imab.onrender.com"
APP_TITLE = "Skin Lesion Research Dashboard"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# API HELPERS
# ------------------------------------------------------------
def get_predict_upload_field():
    """Discover the multipart upload field from FastAPI's OpenAPI schema."""
    try:
        response = requests.get(f"{API_URL}/openapi.json", timeout=5)
        response.raise_for_status()
        schema = response.json()
        request_body = schema["paths"]["/predict"]["post"]["requestBody"]
        multipart = request_body["content"]["multipart/form-data"]["schema"]
        if "$ref" in multipart:
            name = multipart["$ref"].rsplit("/", 1)[-1]
            multipart = schema["components"]["schemas"][name]
        properties = multipart.get("properties", {})
        if properties:
            return next(iter(properties))
    except (requests.RequestException, KeyError, ValueError, TypeError):
        pass
    return "file"


def run_prediction(uploaded_file):
    field_name = get_predict_upload_field()
    response = requests.post(
        f"{API_URL}/predict",
        files={field_name: (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )},
        timeout=180,
    )
    if response.status_code != 200:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        raise RuntimeError(f"API returned HTTP {response.status_code}: {detail}")
    return response.json()


def as_percent(value):
    try:
        number = float(value)
        # API probability values are normally 0-1; support 0-100 too.
        return number * 100 if 0 <= number <= 1 else number
    except (TypeError, ValueError):
        return None


def pretty_label(value):
    return str(value if value is not None else "Unknown").replace("_", " ").title()


def show_disclaimer():
    st.markdown(
        "<div class='notice'><strong>Research prototype — not for clinical use.</strong> "
        "Outputs are experimental, may be inaccurate, and have not been established as a medical "
        "diagnostic service. Do not use them to make medical decisions or delay professional care.</div>",
        unsafe_allow_html=True,
    )


def draw_heatmap(heatmap_data):
    heatmap = np.asarray(heatmap_data, dtype=float)
    if heatmap.ndim != 2 or heatmap.size == 0:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    shown = ax.imshow(heatmap, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title("LFAM feature-attribution map", fontsize=12, pad=12)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.04, label="Relative map value")
    fig.tight_layout()
    return fig


def draw_overlay(original_image, heatmap_data):
    """Make a visual-only overlay from the coarse API heatmap."""
    heatmap = np.asarray(heatmap_data, dtype=float)
    if heatmap.ndim != 2 or heatmap.size == 0:
        return None
    base = original_image.convert("RGB")
    # Upscale the coarse grid for visual comparison only, not as a lesion boundary.
    resized = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255)).resize(
        base.size, Image.Resampling.BILINEAR
    )
    cmap = plt.get_cmap("inferno")
    colored = Image.fromarray(np.uint8(cmap(np.asarray(resized) / 255.0)[..., :3] * 255))
    return Image.blend(base, colored, alpha=0.38)

# ------------------------------------------------------------
# VISUAL DESIGN
# ------------------------------------------------------------
st.markdown("""
<style>
:root { --navy:#123047; --blue:#176b87; --teal:#1b8a8f; --ink:#172b3a; --muted:#637586; --line:#dce6ec; --paper:#f4f8fb; }
.stApp { background:var(--paper); color:var(--ink); }
#MainMenu, footer { visibility:hidden; }
[data-testid="stHeader"] { background:rgba(244,248,251,.88); }
.block-container { padding-top:1.7rem; padding-bottom:2.5rem; max-width:1450px; }
[data-testid="stSidebar"] { background:#102b3e; border-right:1px solid #234458; }
[data-testid="stSidebar"] * { color:#eef6fa; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding-top:5px; padding-bottom:5px; }
[data-testid="stSidebar"] hr { border-color:#315165; }
.hero { background:linear-gradient(120deg,#102b3e 0%,#16536d 62%,#1b8a8f 100%); border-radius:20px; padding:36px 38px; color:white; margin-bottom:24px; box-shadow:0 12px 30px rgba(18,48,71,.12); }
.eyebrow { font-size:11px; font-weight:800; text-transform:uppercase; letter-spacing:1.8px; color:#bce9e8; margin-bottom:12px; }
.hero h1 { color:white; font-size:clamp(28px,3.2vw,42px); line-height:1.13; margin:0 0 12px 0; font-weight:780; }
.hero p { color:#e5f2f7; font-size:16px; line-height:1.7; max-width:820px; margin:0; }
.page-title { color:var(--navy); font-size:32px; font-weight:780; margin:0 0 5px 0; }
.page-subtitle { color:var(--muted); font-size:15px; line-height:1.6; margin-bottom:24px; }
.section-title { color:var(--navy); font-size:22px; font-weight:750; margin:22px 0 5px 0; }
.section-caption { color:var(--muted); font-size:14px; margin-bottom:16px; }
.panel { background:white; border:1px solid var(--line); border-radius:16px; padding:23px; box-shadow:0 4px 15px rgba(18,48,71,.035); margin-bottom:16px; }
.panel-title { color:var(--navy); font-size:18px; font-weight:750; margin-bottom:8px; }
.panel-copy { color:var(--muted); font-size:14px; line-height:1.7; }
.feature { background:white; border:1px solid var(--line); border-radius:15px; padding:21px; height:100%; min-height:160px; }
.feature-icon { font-size:24px; margin-bottom:12px; }
.feature-title { color:var(--navy); font-weight:750; font-size:16px; margin-bottom:7px; }
.feature-copy { color:var(--muted); font-size:13px; line-height:1.65; }
.step { background:white; border:1px solid var(--line); border-radius:12px; padding:16px 10px; text-align:center; min-height:98px; }
.step-num { color:var(--teal); font-size:11px; font-weight:850; letter-spacing:1.2px; margin-bottom:7px; }
.step-label { color:var(--navy); font-size:13px; font-weight:700; line-height:1.4; }
.notice { background:#fff8e8; border:1px solid #efd28a; color:#6c520b; border-radius:12px; padding:15px 18px; font-size:13px; line-height:1.6; margin:18px 0 8px 0; }
.status-pill { display:inline-block; background:#e6f5f4; color:#14676b; padding:5px 10px; border-radius:999px; font-size:11px; font-weight:800; letter-spacing:.6px; text-transform:uppercase; }
div.stButton > button[kind="primary"] { background:var(--blue); border:1px solid var(--blue); border-radius:10px; font-weight:700; min-height:44px; }
div.stButton > button[kind="primary"]:hover { background:var(--navy); border-color:var(--navy); }
[data-testid="stMetric"] { background:white; border:1px solid var(--line); border-radius:13px; padding:16px; }
[data-testid="stMetricLabel"] { color:var(--muted); }
[data-testid="stMetricValue"] { color:var(--navy); }
hr { border-color:var(--line); }

/* Fix faint text in Streamlit information and success messages */
[data-testid="stAlert"],
[data-testid="stAlert"] p,
[data-testid="stAlert"] div,
[data-testid="stAlert"] span {
    color: #243447 !important;
    opacity: 1 !important;
}

/* Improve readability of captions and descriptive text */
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #475569 !important;
    opacity: 1 !important;
}

/* Ensure pale custom message boxes have readable text */
.stApp .notice {
    color: #6c520b !important;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# NAVIGATION
# ------------------------------------------------------------
NAV = ["🏠 Home", "🔬 Analyse", "📊 Fairness Analysis", "ℹ️ About"]
if "navigation_target" in st.session_state:
    st.session_state["navigation_choice"] = st.session_state.pop("navigation_target")

with st.sidebar:
    st.markdown("<div style='font-size:30px;margin-bottom:2px'>🔬</div>", unsafe_allow_html=True)
    st.markdown("## Skin Lesion Research")
    st.caption("CLASSIFICATION DASHBOARD")
    st.divider()
    page = st.radio(
        "Navigate", NAV,
        index=NAV.index(st.session_state.get("navigation_choice", NAV[0]))
        if st.session_state.get("navigation_choice", NAV[0]) in NAV else 0,
        key="navigation_choice", label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**System status**")
    try:
        health_response = requests.get(f"{API_URL}/openapi.json", timeout=1.5)
        api_online = health_response.ok
    except requests.RequestException:
        api_online = False
    st.markdown(
        f"<span class='status-pill'>{'API reachable' if api_online else 'API offline'}</span>",
        unsafe_allow_html=True,
    )
    st.caption("Academic research prototype")

# ------------------------------------------------------------
# HOME
# ------------------------------------------------------------
if page == "🏠 Home":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">Academic research · Image classification</div>
      <h1>Skin Lesion Classification</h1>
      <p>A research dashboard for exploring machine-learning skin lesion classification, experimental model outputs, and feature-attribution visualizations—with a focus on fairness and class imbalance.</p>
    </div>
    """, unsafe_allow_html=True)

    overview, action = st.columns([1.5, 1], gap="large")
    with overview:
        st.markdown("""
        <div class="panel">
          <div class="panel-title">Project overview</div>
          <div class="panel-copy">Skin cancer recognition can be affected by limited representation of darker skin tones, imbalanced datasets, and differences in how lesions appear across people. This interface connects an image upload to the project's classification API and presents the returned outputs for research review.<br><br><b>Current pipeline:</b> image upload → preprocessing and feature extraction → SVM classification → returned prediction and LFAM visualization.</div>
        </div>
        """, unsafe_allow_html=True)
    with action:
        st.markdown("""
        <div class="panel">
          <div class="panel-title">Run an image analysis</div>
          <div class="panel-copy">Upload a JPG, JPEG, or PNG image, submit it to the connected API, and inspect the prediction, returned probabilities, entropy, and available feature-attribution map.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔬 Open image analysis", type="primary", use_container_width=True):
            st.session_state["navigation_target"] = "🔬 Analyse"
            st.rerun()

    st.markdown('<div class="section-title">Dashboard capabilities</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">The main areas of the research interface.</div>', unsafe_allow_html=True)
    cards = st.columns(3, gap="medium")
    feature_cards = [
        ("🖼️", "Image analysis", "Submit a lesion image to the connected FastAPI endpoint and view the returned classification."),
        ("📈", "Model outputs", "Inspect the predicted label, class probability values, Shannon entropy, and technical metadata."),
        ("⚖️", "Fairness research", "Review the planned evaluation of performance, class imbalance, and representation across skin-tone groups."),
    ]
    for col, (icon, title, copy) in zip(cards, feature_cards):
        with col:
            st.markdown(f"<div class='feature'><div class='feature-icon'>{icon}</div><div class='feature-title'>{title}</div><div class='feature-copy'>{copy}</div></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Classification workflow</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">High-level view of the research pipeline. The API remains responsible for its actual processing steps.</div>', unsafe_allow_html=True)
    workflow = [("01", "Image upload"), ("02", "Preprocessing"), ("03", "ResNet-50 features"), ("04", "SVM classification"), ("05", "Prediction + LFAM")]
    cols = st.columns(5, gap="small")
    for col, (number, label) in zip(cols, workflow):
        with col:
            st.markdown(f"<div class='step'><div class='step-num'>{number}</div><div class='step-label'>{label}</div></div>", unsafe_allow_html=True)
    show_disclaimer()

# ------------------------------------------------------------
# ANALYSE
# ------------------------------------------------------------
elif page == "🔬 Analyse":
    st.markdown('<div class="page-title">Image analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Upload an image and send it to the connected classification API. Results shown here are the API response, not a clinical diagnosis.</div>', unsafe_allow_html=True)

    upload_col, preview_col = st.columns([1, 1], gap="large")
    with upload_col:
        st.markdown("<div class='panel'><div class='panel-title'>📤 Upload a lesion image</div><div class='panel-copy'>Use a clear JPG, JPEG, or PNG image. For a meaningful research evaluation, document the source and relevant dataset metadata separately.</div></div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key="lesion_upload")
        if uploaded_file:
            st.success(f"Ready to analyse: {uploaded_file.name}")
        else:
            st.info("No image selected yet.")
    original_image = None
    with preview_col:
        st.markdown("<div class='panel'><div class='panel-title'>🖼️ Image preview</div><div class='panel-copy'>The selected image is displayed here before submission.</div>", unsafe_allow_html=True)
        if uploaded_file:
            try:
                original_image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
                st.image(original_image, caption="Uploaded image", use_column_width=True)
            except Exception:
                st.error("This file could not be opened as an image. Please choose another JPG, JPEG, or PNG file.")
        else:
            st.caption("Your image preview will appear here.")
        st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file and original_image is not None:
        if st.button("🔍 Analyse image", type="primary", use_container_width=True):
            with st.spinner("Sending image to the classification API. This may take a little while..."):
                try:
                    st.session_state["last_prediction"] = run_prediction(uploaded_file)
                    st.session_state["last_image_bytes"] = uploaded_file.getvalue()
                    st.session_state["last_image_name"] = uploaded_file.name
                    st.session_state.pop("prediction_error", None)
                except requests.exceptions.ConnectionError:
                    st.session_state["prediction_error"] = f"Could not connect to FastAPI at {API_URL}. Start the API server in its own terminal and try again."
                    st.session_state.pop("last_prediction", None)
                except requests.exceptions.Timeout:
                    st.session_state["prediction_error"] = "The API took too long to respond. Check the API terminal and try again."
                    st.session_state.pop("last_prediction", None)
                except requests.RequestException as exc:
                    st.session_state["prediction_error"] = f"The API request failed: {exc}"
                    st.session_state.pop("last_prediction", None)
                except RuntimeError as exc:
                    st.session_state["prediction_error"] = str(exc)
                    st.session_state.pop("last_prediction", None)

    if st.session_state.get("prediction_error"):
        st.error(st.session_state["prediction_error"])
    result = st.session_state.get("last_prediction")
    result_image_bytes = st.session_state.get("last_image_bytes")
    if result and result_image_bytes:
        st.markdown('<div class="section-title">Classification results</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">These values are shown as returned by the connected API.</div>', unsafe_allow_html=True)
        label = pretty_label(result.get("predicted_label", result.get("predicted_class", "Unknown")))
        probs = result.get("named_class_probabilities") or result.get("class_probabilities") or {}
        predicted_probability = None
        if isinstance(probs, dict):
            predicted_probability = probs.get(result.get("predicted_label"))
            if predicted_probability is None:
                predicted_probability = probs.get(str(result.get("predicted_class")))
        probability_percent = as_percent(predicted_probability)
        entropy = result.get("shannon_entropy")
        metric_cols = st.columns(3, gap="medium")
        with metric_cols[0]:
            st.metric("Predicted label", label)
        with metric_cols[1]:
            st.metric("Predicted-class probability", f"{probability_percent:.2f}%" if probability_percent is not None else "Not provided")
        with metric_cols[2]:
            try:
                st.metric("Shannon entropy", f"{float(entropy):.4f}")
            except (TypeError, ValueError):
                st.metric("Shannon entropy", "Not provided")

        if isinstance(probs, dict) and probs:
            st.markdown('<div class="section-title">Class probabilities</div>', unsafe_allow_html=True)
            prob_cols = st.columns(max(1, min(len(probs), 3)), gap="medium")
            for i, (class_name, raw_value) in enumerate(probs.items()):
                pct = as_percent(raw_value)
                with prob_cols[i % len(prob_cols)]:
                    st.markdown("<div class='panel'>", unsafe_allow_html=True)
                    st.markdown(f"**{html.escape(pretty_label(class_name))}**")
                    if pct is not None:
                        st.progress(min(max(pct / 100, 0), 1))
                        st.write(f"{pct:.2f}%")
                    else:
                        st.write("Value not available")
                    st.markdown("</div>", unsafe_allow_html=True)
            st.caption("Model probability outputs are not the same as medically validated confidence estimates. Shannon entropy is a separate metric.")

        interpretability = result.get("interpretability") or {}
        heatmap_data = interpretability.get("heatmap_data")
        if interpretability.get("available") and heatmap_data:
            st.markdown('<div class="section-title">Image and LFAM visualization</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-caption">The map is a coarse experimental feature-attribution visualization; it is not a lesion outline or a verified explanation of clinical reasoning.</div>', unsafe_allow_html=True)
            try:
                base_image = Image.open(io.BytesIO(result_image_bytes)).convert("RGB")
                overlay = draw_overlay(base_image, heatmap_data)
                original_col, overlay_col, map_col = st.columns(3, gap="medium")
                with original_col:
                    st.markdown("**Original image**")
                    st.image(base_image, use_column_width=True)
                with overlay_col:
                    st.markdown("**Experimental overlay**")
                    if overlay is not None:
                        st.image(overlay, use_column_width=True)
                with map_col:
                    st.markdown("**LFAM heatmap**")
                    fig = draw_heatmap(heatmap_data)
                    if fig is not None:
                        st.pyplot(fig, clear_figure=True)
                        plt.close(fig)
                st.caption(interpretability.get("validation_notice", "Experimental feature-attribution map; not clinically validated."))
                st.caption(interpretability.get("overlay_note", "The overlay is for visual exploration only and should not be treated as a lesion boundary."))
            except (TypeError, ValueError, OSError) as exc:
                st.info(f"The API returned interpretability data, but it could not be rendered: {exc}")
        else:
            st.info("The API did not return an available LFAM heatmap for this image.")

        with st.expander("Technical details from API"):
            details = {
                "Filename": result.get("filename", st.session_state.get("last_image_name", "—")),
                "Predicted class ID": result.get("predicted_class", "—"),
                "Feature count": result.get("feature_count", "—"),
                "PCA components": result.get("pca_components", "—"),
                "PCA used": result.get("pca_used", "—"),
                "Decision score": result.get("decision_score", "—"),
                "Probability concentration score (%)": result.get("probability_concentration_score_percent", "—"),
                "Class mapping verified": result.get("class_mapping_verified", "Not provided"),
            }
            st.json(details)
        research_notice = result.get("research_notice")
        if research_notice:
            st.info(str(research_notice))
    show_disclaimer()

# ------------------------------------------------------------
# FAIRNESS ANALYSIS
# ------------------------------------------------------------
elif page == "📊 Fairness Analysis":
    st.markdown('<div class="page-title">Fairness analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Explore dataset representation, class imbalance, and the evidence needed to compare fairness across Fitzpatrick skin-type groups.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="panel"><div class="panel-title">Why fairness is being evaluated</div><div class="panel-copy">A strong overall score can hide weaker detection of malignant lesions in particular skin-tone groups. This project compares six class-imbalance mitigation approaches against an unbalanced baseline and evaluates both predictive performance and disparities between Fitzpatrick skin-type groups.</div></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Dataset composition reported in the research proposal</div>', unsafe_allow_html=True)
    st.caption("These graphs describe the documented dataset counts, not model accuracy or fairness outcomes.")
    class_counts = pd.DataFrame({
        "Dataset": ["DDI", "DDI", "MRA-MIDAS", "MRA-MIDAS"],
        "Diagnostic class": ["Benign / non-malignant", "Malignant", "Benign / non-malignant", "Malignant"],
        "Images": [485, 171, 1655, 412],
    })
    tone_counts = pd.DataFrame({
        "Fitzpatrick group": ["I–II", "III–IV", "V–VI"],
        "DDI records": [208, 241, 207],
    })
    chart_left, chart_right = st.columns(2, gap="large")
    with chart_left:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        pivot = class_counts.pivot(index="Dataset", columns="Diagnostic class", values="Images")
        pivot = pivot[["Benign / non-malignant", "Malignant"]]
        pivot.plot(kind="bar", ax=ax, color=["#4f9b91", "#d66b6b"])
        ax.set_title("Diagnostic class counts")
        ax.set_xlabel("")
        ax.set_ylabel("Number of records")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="Class", frameon=False)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    with chart_right:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        bars = ax.bar(tone_counts["Fitzpatrick group"], tone_counts["DDI records"], color=["#4f7894", "#398b83", "#1b5a68"])
        ax.set_title("DDI records by skin-tone group")
        ax.set_xlabel("Fitzpatrick skin-type group")
        ax.set_ylabel("Number of records")
        ax.bar_label(bars, padding=3)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    st.markdown('<div class="section-title">Research design</div>', unsafe_allow_html=True)
    method_cols = st.columns(3, gap="medium")
    for col, (title, copy) in zip(method_cols, [
        ("Baseline", "Evaluate the model trained on the original imbalanced training data."),
        ("Mitigation", "Compare random oversampling, random undersampling, image augmentation, SMOTE, ADASYN, and SVM class weighting."),
        ("Fairness metric", "Use Equal Opportunity Difference (EOD) to examine differences in malignant-case true-positive rates between skin-tone groups."),
    ]):
        with col:
            st.markdown(f"<div class='feature'><div class='feature-title'>{title}</div><div class='feature-copy'>{copy}</div></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">How fairness will be measured</div>', unsafe_allow_html=True)
    st.markdown(r"""
    <div class="panel"><div class="panel-title">Equal Opportunity Difference (EOD)</div><div class="panel-copy">True-positive rate (TPR) measures the proportion of actual malignant cases correctly identified by the model. EOD compares TPR between the groups being studied. A value closer to zero indicates a smaller sensitivity gap for the chosen comparison; the sign indicates which group has the higher TPR. Always show the group definitions and sample counts with the result.</div></div>
    """, unsafe_allow_html=True)
    st.latex(r"TPR = \frac{TP}{TP + FN} \qquad EOD = TPR_{group\ A} - TPR_{group\ B}")

    st.markdown("<div class='section-title'>Evaluation Results </div>", unsafe_allow_html=True)
    evaluation_dir = Path(__file__).resolve().parent / "evaluation_files"
    expected_csvs = [
        "overall_final_metrics.csv",
        "stratified_final_metrics.csv",
        "overall_eod.csv",
        "shannon_overall_entropies.csv",
    ]
    results_frames = []
    missing_csvs = []
    for csv_name in expected_csvs:
        csv_path = evaluation_dir / csv_name
        if csv_path.is_file():
            try:
                results_frames.append((csv_name, pd.read_csv(csv_path)))
            except Exception as exc:
                st.error(f"Could not read {csv_name}: {exc}")
        else:
            missing_csvs.append(csv_name)

    def metric_mean(series):
        # Evaluation files store averages as strings such as '0.566 ± 0.034'.
        return pd.to_numeric(
            series.astype(str).str.extract(r"^\s*([-+]?(?:\d*\.)?\d+)")[0],
            errors="coerce",
        )

    if results_frames:
        for results_name, results_df in results_frames:
            st.markdown(f"### {results_name}")
            try:
                if results_df.empty:
                    st.info("This CSV contains no data rows.")
                    continue
                st.dataframe(results_df, use_container_width=True)
                lower_cols = {str(c).strip().lower().replace("_", " "): c for c in results_df.columns}
                method_col = next((lower_cols[k] for k in ["method", "resampling method", "strategy"] if k in lower_cols), None)
                group_col = next((lower_cols[k] for k in ["skin tone group", "skin tone", "fitzpatrick group", "group"] if k in lower_cols), None)
                eod_col = next((c for c in results_df.columns if "eod" in str(c).lower() or "equal opportunity" in str(c).lower()), None)
                entropy_col = next((c for c in results_df.columns if "entropy" in str(c).lower()), None)
                # Convert mean ± standard deviation strings to the mean for plotting.
                metric_cols = []
                for c in results_df.columns:
                    name = str(c).lower().replace("_", " ")
                    if any(term in name for term in ["recall", "tpr", "balanced", "accuracy", "f1", "specificity", "precision", "roc auc", "pr auc", "eod", "equal opportunity", "entropy"]):
                        numeric = metric_mean(results_df[c])
                        if numeric.notna().any():
                            results_df[f"{c} (mean)"] = numeric
                            metric_cols.append(f"{c} (mean)")
                if not metric_cols:
                    st.info("The file loaded, but no recognised numeric evaluation metrics were found.")
                    continue
                selected_metric = st.selectbox(
                    f"Metric to chart — {results_name}", metric_cols,
                    key=f"metric_{results_name}",
                )
                if method_col and group_col:
                    chart_data = results_df.pivot_table(index=method_col, columns=group_col, values=selected_metric, aggfunc="mean")
                    fig, ax = plt.subplots(figsize=(10, 4.8))
                    chart_data.plot(kind="bar", ax=ax)
                    ax.set_title(f"{selected_metric.replace(' (mean)', '')} by method and skin-tone group")
                    ax.set_xlabel("Evaluation method")
                    ax.set_ylabel("Mean value")
                    ax.tick_params(axis="x", rotation=25)
                    ax.legend(title="Skin-tone group", frameon=False)
                    fig.tight_layout()
                    st.pyplot(fig, clear_figure=True)
                    plt.close(fig)
                elif method_col:
                    chart_data = results_df.groupby(method_col)[selected_metric].mean().dropna()
                    if not chart_data.empty:
                        fig, ax = plt.subplots(figsize=(9, 4.5))
                        chart_data.plot(kind="bar", ax=ax, color="#398b83")
                        ax.set_title(f"{selected_metric.replace(' (mean)', '')} by method")
                        ax.set_xlabel("Evaluation method")
                        ax.set_ylabel("Mean value")
                        ax.tick_params(axis="x", rotation=25)
                        fig.tight_layout()
                        st.pyplot(fig, clear_figure=True)
                        plt.close(fig)
                else:
                    st.info("This file has metrics but no recognised method column, so a method comparison cannot be plotted.")
                if eod_col is not None:
                    st.caption("EOD is a difference between true-positive rates; values nearer zero indicate a smaller measured gap for the specific comparison. Check the notebook definition before interpreting its sign.")
                if entropy_col is not None:
                    st.caption("Shannon entropy describes uncertainty in the output distribution; it is not a fairness score or a medical confidence guarantee.")
            except Exception as exc:
                st.error(f"Could not process {results_name}: {exc}")
    else:
        st.warning("No evaluation CSV files were found in the `evaluation_files` folder next to `app.py`. Create that folder and place the four project CSV files inside it, then refresh the page.")
    if missing_csvs:
        with st.expander("Missing evaluation files"):
            st.write("The following expected files were not found:")
            for missing_name in missing_csvs:
                st.write(f"- `{missing_name}`")
            st.caption(f"Expected folder: {evaluation_dir}")
    show_disclaimer()

# ------------------------------------------------------------
# ABOUT
# ------------------------------------------------------------
elif page == "ℹ️ About":
    st.markdown('<div class="page-title">About the project</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Research context, aims, questions, methodology, and responsible interpretation.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="panel"><div class="panel-title">Project title</div><div class="panel-copy"><b>Comparative Study of Resampling Techniques for Addressing Class Imbalance in Fitzpatrick Skin Type Groups: Applications in Skin Cancer Image Classification</b></div></div>
    <div class="panel"><div class="panel-title">Research problem</div><div class="panel-copy">Dermatology image datasets may contain unequal numbers of benign and malignant lesions and unequal representation of Fitzpatrick skin-type groups. A model can achieve reasonable overall performance while missing more malignant lesions in some groups. This project investigates whether class-imbalance mitigation techniques can improve classification performance while reducing sensitivity disparities across skin-tone groups.</div></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Research aim</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="panel"><div class="panel-copy">To investigate how different class-imbalance mitigation techniques affect the classification performance and fairness of skin cancer image classification models across Fitzpatrick skin-type groups, using the DDI and MRA-MIDAS datasets.</div></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Research Objectives</div>', unsafe_allow_html=True)
    questions = [
        "To analyse and visualise the class imbalance.",
        "To assess the impact of the different resampling techniques on model fairness and classification performance across Fitzpatrick skin type groups.",
        "To apply resampling methods in mitigating the effects of class imbalance prior to model training and observe the label ratios",
    ]
    for i, question in enumerate(questions, start=1):
        st.markdown(f"<div class='panel'><div class='panel-title'>Research Objective {i}</div><div class='panel-copy'>{question}</div></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Datasets and methodology</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="panel"><div class="panel-title">Data sources</div><div class="panel-copy"><b>DDI:</b> 656 records in the project proposal, including 485 non-malignant and 171 malignant cases.<br><b>MRA-MIDAS:</b> 2,077 records in the project proposal, including 1,655 non-melanoma and 412 melanoma cases. These are proposal-reported counts; the final cleaned analysis dataset may differ.</div></div>
    <div class="panel"><div class="panel-title">Proposed modelling workflow</div><div class="panel-copy"><b>Image and metadata preparation → data split → training-only imbalance mitigation → ResNet-50 feature extraction → SVM classification → evaluation on an untouched test set.</b><br><br>The deployed prototype currently sends an uploaded image to the existing FastAPI backend. The backend performs the available inference pipeline and returns a prediction, probability-related values, Shannon entropy, and an experimental LFAM map.</div></div>
    <div class="panel"><div class="panel-title">Methods compared in the study</div><div class="panel-copy">Unbalanced baseline, random oversampling, random undersampling, image augmentation, SMOTE, ADASYN, and cost-sensitive SVM class weighting. Resampling should be applied only to training data to avoid leakage into evaluation.</div></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Understanding the app outputs</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="panel"><div class="panel-copy"><b>Predicted label:</b> the class returned by the API; class mapping must be verified by the model developer.<br><br><b>Class probabilities:</b> numerical model outputs; they should not automatically be interpreted as calibrated medical certainty.<br><br><b>Shannon entropy:</b> a measure derived from the output probability distribution, shown separately from the predicted-class probability.<br><br><b>LFAM visualization:</b> an experimental coarse map for research exploration, not a verified lesion boundary or a clinically validated explanation.</div></div>
    <div class="panel"><div class="panel-title">Responsible use</div><div class="panel-copy">This is an academic research prototype, not a clinical diagnostic system. It has not been clinically validated and must not be used to diagnose skin cancer, guide treatment, or rule out disease. Concerning or changing lesions should be assessed by a qualified healthcare professional.</div></div>
    """, unsafe_allow_html=True)
    show_disclaimer()
