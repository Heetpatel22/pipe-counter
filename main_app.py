import tempfile
from importlib import import_module
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from gemini_process import process_industrial_pipes


def get_gemini_analysis(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name
    try:
        return process_industrial_pipes(temp_path)
    except Exception as e:
        return f"Failed to run Gemini analysis: {e}"
    finally:
        Path(temp_path).unlink(missing_ok=True)


BASE_DIR = Path(__file__).resolve().parent

PIPE_STREAMS = {
    "hollow": {
        "label": "Hollow Pipe",
        "folder": BASE_DIR / "hollow_specific",
        "weights": BASE_DIR / "hollow_specific" / "best_hollow.onnx",
        "module": "hollow_specific.type_breaker",
        "default_conf": 0.25,
        "accent": "#0f766e",
    },
    "solid": {
        "label": "Solid Pipe",
        "folder": BASE_DIR / "solid_specific",
        "weights": BASE_DIR / "solid_specific" / "best_solid.onnx",
        "module": "solid_specific.type_breaker",
        "default_conf": 0.60,
        "accent": "#1d4ed8",
    },
}


st.set_page_config(
    layout="wide",
    page_title="Industrial Pipe Counter",
    page_icon="IPC",
)

st.markdown(
    """
    <style>
        :root {
            --bg: #f5f7fb;
            --panel: #ffffff;
            --panel-soft: #f8fafc;
            --ink: #172033;
            --muted: #65758b;
            --line: #dbe3ef;
            --accent: #0f766e;
            --danger: #b91c1c;
            --ok: #166534;
        }

        .stApp {
            background:
                linear-gradient(180deg, rgba(15, 118, 110, 0.08), rgba(245, 247, 251, 0) 280px),
                var(--bg);
            color: var(--ink);
        }

        .block-container {
            max-width: 1400px;
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }

        [data-testid="stSidebar"] {
            background: #111827;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] * {
            color: #f8fafc;
        }

        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: #cbd5e1;
        }

        .dash-header {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 1.5rem;
            padding: 1.2rem 0 1.1rem;
            border-bottom: 1px solid var(--line);
            margin-bottom: 1.1rem;
        }

        .eyebrow {
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .dash-title {
            color: var(--ink);
            font-size: clamp(2rem, 4vw, 3.5rem);
            line-height: 1;
            font-weight: 850;
            letter-spacing: 0;
            margin: 0;
        }

        .dash-subtitle {
            color: var(--muted);
            max-width: 780px;
            font-size: 1rem;
            line-height: 1.55;
            margin: 0.75rem 0 0;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 132px;
            padding: 0.55rem 0.8rem;
            border-radius: 999px;
            background: #dcfce7;
            color: var(--ok);
            font-size: 0.86rem;
            font-weight: 750;
            border: 1px solid #bbf7d0;
            white-space: nowrap;
        }

        .section-panel {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 16px 40px rgba(15, 23, 42, 0.06);
            margin-bottom: 0.75rem;
        }

        .panel-title {
            color: var(--ink);
            font-size: 1rem;
            font-weight: 800;
            margin: 0 0 0.2rem;
        }

        .panel-note {
            color: var(--muted);
            font-size: 0.88rem;
            margin: 0;
        }

        .pipe-choice {
            border: 1px solid rgba(203, 213, 225, 0.22);
            border-radius: 8px;
            padding: 0.8rem 0.85rem;
            margin: 0.45rem 0;
            background: rgba(255, 255, 255, 0.05);
        }

        .pipe-choice strong {
            display: block;
            color: #f8fafc;
            font-size: 0.95rem;
            margin-bottom: 0.2rem;
        }

        .pipe-choice span {
            color: #94a3b8;
            font-size: 0.78rem;
        }

        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
            margin-bottom: 1rem;
            max-width: 460px;
        }

        .kpi-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.9rem 1rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
        }

        .kpi-label {
            color: var(--muted);
            font-size: 0.77rem;
            font-weight: 750;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.35rem;
        }

        .kpi-value {
            color: var(--ink);
            font-size: 1.9rem;
            line-height: 1;
            font-weight: 850;
        }

        .kpi-foot {
            color: var(--muted);
            font-size: 0.78rem;
            margin-top: 0.45rem;
        }

        .invalid-result {
            color: var(--danger);
            font-size: 1rem;
            font-weight: 750;
            margin: 0.4rem 0 1rem;
        }

        .empty-state {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 420px;
            border: 1px dashed #b8c4d6;
            border-radius: 8px;
            background: var(--panel-soft);
            color: var(--muted);
            text-align: center;
            padding: 2rem;
        }

        .empty-state strong {
            display: block;
            color: var(--ink);
            font-size: 1.15rem;
            margin-bottom: 0.35rem;
        }

        @media (max-width: 900px) {
            .dash-header {
                align-items: flex-start;
                flex-direction: column;
            }

            .kpi-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def select_pipe_type(pipe_type):
    st.session_state.pipe_type = pipe_type
    st.session_state.pick_hollow = pipe_type == "hollow"
    st.session_state.pick_solid = pipe_type == "solid"


def prepare_uploaded_image(uploaded_file, apply_glare_filter):
    original = Image.open(uploaded_file).convert("RGB")

    if not apply_glare_filter:
        return original, "Raw image"

    opencv_img = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)
    lab_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl_channel = clahe.apply(l_channel)
    merged_lab = cv2.merge((cl_channel, a_channel, b_channel))
    processed_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(processed_rgb), "CLAHE filtered"


def run_selected_stream(pipe_type, image, conf_threshold, iou_threshold):
    stream = PIPE_STREAMS[pipe_type]
    breaker = import_module(stream["module"])

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image:
        image.save(temp_image.name)
        temp_path = temp_image.name

    try:
        return breaker.analyze_inventory(
            temp_path,
            str(stream["weights"]),
            conf=conf_threshold,
            iou=iou_threshold,
        )
    finally:
        Path(temp_path).unlink(missing_ok=True)


if "pipe_type" not in st.session_state:
    st.session_state.pipe_type = "hollow"
    st.session_state.pick_hollow = True
    st.session_state.pick_solid = False


st.markdown(
    """
    <div class="dash-header">
        <div>
            <div class="eyebrow">Computer vision dashboard</div>
            <h1 class="dash-title">Industrial Pipe Counter</h1>
            <p class="dash-subtitle">
                Select the pipe stream, upload an inspection image, and run the matching ONNX
                model with its own type-breaker rules.
            </p>
        </div>
        <div class="status-pill">Routing ready</div>
    </div>
    """,
    unsafe_allow_html=True,
)


st.sidebar.markdown("## Pipe Stream")
st.sidebar.caption("Only one stream is active at a time.")

st.sidebar.markdown(
    """
    <div class="pipe-choice">
        <strong>Hollow Pipe</strong>
        <span>Uses hollow_specific/best_hollow.onnx and hollow type breaker.</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.checkbox(
    "Tick hollow pipe",
    key="pick_hollow",
    on_change=select_pipe_type,
    args=("hollow",),
)

st.sidebar.markdown(
    """
    <div class="pipe-choice">
        <strong>Solid Pipe</strong>
        <span>Uses solid_specific/best_solid.onnx and solid type breaker.</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.checkbox(
    "Tick solid pipe",
    key="pick_solid",
    on_change=select_pipe_type,
    args=("solid",),
)

active_stream = PIPE_STREAMS[st.session_state.pipe_type]

st.sidebar.markdown("---")
st.sidebar.markdown("## Model Controls")
conf_threshold = st.sidebar.slider(
    "Confidence cutoff",
    0.05,
    1.0,
    float(active_stream["default_conf"]),
    0.05,
    key=f"conf_{st.session_state.pipe_type}",
)
iou_threshold = st.sidebar.slider("IOU cutoff", 0.05, 1.0, 0.45, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("## Image Preparation")
apply_glare_filter = st.sidebar.toggle("Glare reduction (CLAHE)", value=False)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    **Active stream**  
    Pipe type: `{active_stream["label"]}`  
    Model: `{active_stream["weights"].name}`  
    Type breaker: `{active_stream["folder"].name}/type_breaker.py`
    """
)

missing_weights = not active_stream["weights"].exists()

left_col, right_col = st.columns([0.92, 1.08], gap="large")

with left_col:
    st.markdown(
        """
        <div class="section-panel">
            <p class="panel-title">Source Image</p>
            <p class="panel-note">Add a JPG or PNG from the inspection site.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Browse your computer or drop an image here...",
        type=["jpg", "jpeg", "png"],
    )

    prepared_image = None
    image_mode = "Waiting"

    if uploaded_file is not None:
        prepared_image, image_mode = prepare_uploaded_image(uploaded_file, apply_glare_filter)
        st.image(prepared_image, caption=f"Uploaded image - {image_mode.lower()}", use_container_width=True)
    else:
        st.markdown(
            """
            <div class="empty-state">
                <div>
                    <strong>No image loaded</strong>
                    Drop an inspection photo above to start the selected stream.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with right_col:
    st.markdown(
        """
        <div class="section-panel">
            <p class="panel-title">Detection Analytics</p>
            <p class="panel-note">Results from the selected model and type-breaker flow.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if missing_weights:
        st.error(f"Missing model file: {active_stream['weights']}")
    elif prepared_image is not None:
        # Cache Gemini analysis by filename and size to prevent repeated API calls
        cache_key = f"gemini_{uploaded_file.name}_{uploaded_file.size}"
        if cache_key not in st.session_state:
            with st.spinner("Running Gemini glare-restoration analysis..."):
                st.session_state[cache_key] = get_gemini_analysis(uploaded_file)
        
        gemini_analysis = st.session_state[cache_key]

        with st.spinner(f"Running {active_stream['label']} stream..."):
            report = run_selected_stream(
                st.session_state.pipe_type,
                prepared_image,
                conf_threshold,
                iou_threshold,
            )

        if not report["passed"]:
            st.markdown(
                f"""
                <div class="invalid-result">
                    Invalid detection. Please choose the correct pipe type for this batch.
                </div>
                <p class="panel-note">{report["reason"]}</p>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="kpi-grid">
                    <div class="kpi-card">
                        <div class="kpi-label">Count</div>
                        <div class="kpi-value">{report["count"]}</div>
                        <div class="kpi-foot">{active_stream["label"]} accepted</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-label">Confidence</div>
                        <div class="kpi-value">{report["avg_confidence"]:.1f}%</div>
                        <div class="kpi-foot">Mean model score</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if report["annotated_image"] is not None:
                st.image(report["annotated_image"], caption="Processed detection output", use_container_width=True)
    else:
        st.markdown(
            """
            <div class="empty-state">
                <div>
                    <strong>Awaiting analysis</strong>
                    Select hollow or solid in the sidebar, then upload an image.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
