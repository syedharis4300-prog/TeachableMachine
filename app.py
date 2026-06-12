import streamlit as st
import requests
import pandas as pd
import altair as alt
import time
import io
from PIL import Image

# Backend Service Configuration
BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Neural Studio - Full-Stack AI Trainer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom premium CSS injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global Overrides */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(30, 20, 70, 0.4) 0%, rgba(10, 10, 20, 0) 90%), #0e0c15;
    }
    
    /* Header Area */
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 20px;
        background: rgba(21, 19, 36, 0.8);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    .logo-area {
        display: flex;
        align-items: center;
        gap: 15px;
    }
    
    .logo-icon {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #6e44ff, #b249f8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: pulse 3s infinite alternate;
    }
    
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(to right, #ffffff, #c2bdf4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .header-subtitle {
        font-size: 0.95rem;
        color: #8b8a9f;
        margin: 0;
    }
    
    /* Glassmorphism Cards */
    .card-container {
        background: rgba(21, 19, 36, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 24px 0 rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .card-container:hover {
        border-color: rgba(110, 68, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(110, 68, 255, 0.15);
    }
    
    .card-header-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        padding-bottom: 8px;
    }
    
    /* Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .status-online {
        background: rgba(16, 185, 129, 0.1);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    .status-offline {
        background: rgba(239, 68, 68, 0.1);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
    
    .status-training {
        background: rgba(245, 158, 11, 0.1);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.2);
        animation: pulse 1.5s infinite alternate;
    }
    
    /* Confidence Bars */
    .confidence-item {
        background: rgba(15, 14, 23, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 10px;
    }
    
    .confidence-meta {
        display: flex;
        justify-content: space-between;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 6px;
    }
    
    .confidence-label {
        color: #ffffff;
    }
    
    .confidence-value {
        color: #b249f8;
    }
    
    .progress-track {
        background: #0f0e17;
        border-radius: 6px;
        height: 10px;
        width: 100%;
        overflow: hidden;
    }
    
    .progress-fill {
        background: linear-gradient(90deg, #6e44ff, #b249f8);
        height: 100%;
        border-radius: 6px;
        transition: width 0.4s cubic-bezier(0.1, 0.8, 0.3, 1.0);
    }
    
    /* Animations */
    @keyframes pulse {
        0% { transform: scale(1); opacity: 0.9; }
        100% { transform: scale(1.03); opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Helper function to check API health
def check_api_health():
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if response.status_code == 200:
            return True, response.json()
    except Exception:
        pass
    return False, {}

# Fetch classes and sample counts from backend
def fetch_classes():
    try:
        response = requests.get(f"{BACKEND_URL}/classes", timeout=2)
        if response.status_code == 200:
            data = response.json()
            return data.get("classes", []), data.get("sample_counts", {})
    except Exception:
        pass
    return [], {}

# Initialize session state for tracking training status
if "training_active" not in st.session_state:
    st.session_state.training_active = False

# Render Header Area
is_healthy, health_data = check_api_health()
backend_badge = (
    f'<span class="status-badge status-online"><span style="width:8px;height:8px;border-radius:50%;background:#10b981;display:inline-block;"></span>Backend Online ({health_data.get("device", "CPU")})</span>'
    if is_healthy else
    '<span class="status-badge status-offline"><span style="width:8px;height:8px;border-radius:50%;background:#ef4444;display:inline-block;"></span>Backend Offline</span>'
)

st.markdown(f"""
<div class="header-container">
    <div class="logo-area">
        <div class="logo-icon">🧠</div>
        <div>
            <h1 class="header-title">Neural Studio</h1>
            <p class="header-subtitle">A professional decoupled client-server Teachable Machine clone powered by Streamlit and FastAPI.</p>
        </div>
    </div>
    <div>
        {backend_badge}
    </div>
</div>
""", unsafe_allow_html=True)

if not is_healthy:
    st.error("🔌 Unable to connect to the FastAPI backend. Please ensure the backend server is running on http://127.0.0.1:8000.")
    st.info("Run the orchestrator command `python run.py` to start both frontend and backend services automatically.")
    st.stop()

# Load current classes and count metadata
classes, sample_counts = fetch_classes()

# Main UI layout grid (3 Columns)
col1, col2, col3 = st.columns([1.2, 1.0, 1.2], gap="medium")

# ==============================================================================
# COLUMN 1: DATA GATHERING
# ==============================================================================
with col1:
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown('<div class="card-header-title">📁 1. Training Classes</div>', unsafe_allow_html=True)
    
    # Add new class control
    with st.form("add_class_form", clear_on_submit=True):
        new_class_name = st.text_input("Create a new class label:", placeholder="e.g. Rock, Paper, Scissors")
        submit_class = st.form_submit_button("＋ Add Class")
        if submit_class and new_class_name:
            clean_name = new_class_name.strip()
            if clean_name:
                try:
                    res = requests.post(f"{BACKEND_URL}/classes", params={"class_name": clean_name})
                    if res.status_code == 200:
                        st.success(f"Class '{clean_name}' added successfully!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(res.json().get("detail", "Error adding class"))
                except Exception as e:
                    st.error(f"Request failed: {e}")

    # Webcam capture station
    st.markdown("---")
    st.markdown("##### 📷 Webcam Capture Station")
    st.caption("Capture image samples and route them instantly to one of your classes below.")
    webcam_img = st.camera_input("Capture Frame", key="capture_webcam")
    
    if webcam_img and classes:
        # Provide routing buttons for each class
        st.write("Assign captured image to class:")
        route_cols = st.columns(min(len(classes), 3))
        for idx, class_name in enumerate(classes):
            col_target = route_cols[idx % 3]
            if col_target.button(f"📥 {class_name}", key=f"route_{class_name}"):
                img_bytes = webcam_img.getvalue()
                try:
                    res = requests.post(
                        f"{BACKEND_URL}/upload/{class_name}",
                        files={"files": (f"webcam_{int(time.time())}.jpg", img_bytes, "image/jpeg")}
                    )
                    if res.status_code == 200:
                        st.toast(f"Saved webcam frame to '{class_name}'!")
                        # Wait a bit then rerun to refresh counts
                        time.sleep(0.3)
                        st.rerun()
                    else:
                        st.error(res.json().get("detail", "Failed to save frame"))
                except Exception as e:
                    st.error(f"Failed: {e}")
    elif webcam_img and not classes:
        st.warning("Please configure at least one class label to route webcam frames.")

    # Class details and file uploaders
    st.markdown("---")
    st.markdown("##### Configured Classes")
    if not classes:
        st.info("No classes configured yet. Create a class label above to get started.")
        
    for class_name in classes:
        count = sample_counts.get(class_name, 0)
        # Unique expander card for each class
        with st.expander(f"🏷️ **{class_name}** ({count} samples)", expanded=(count == 0)):
            # Add file upload to this class
            uploaded_files = st.file_uploader(
                f"Upload images for {class_name}:",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
                key=f"uploader_{class_name}"
            )
            
            if uploaded_files:
                files_payload = []
                for uf in uploaded_files:
                    files_payload.append(("files", (uf.name, uf.getvalue(), uf.type)))
                
                if st.button(f"Upload {len(uploaded_files)} files to {class_name}", key=f"btn_up_{class_name}"):
                    with st.spinner("Uploading samples..."):
                        try:
                            res = requests.post(f"{BACKEND_URL}/upload/{class_name}", files=files_payload)
                            if res.status_code == 200:
                                st.success(f"Successfully uploaded {len(uploaded_files)} images!")
                                time.sleep(0.5)
                                st.rerun()
                            else:
                                st.error(res.json().get("detail", "Upload failed"))
                        except Exception as e:
                            st.error(f"Request failed: {e}")
            
            # Delete class option
            if st.button(f"🗑️ Delete Class {class_name}", key=f"del_{class_name}", type="secondary"):
                try:
                    res = requests.delete(f"{BACKEND_URL}/classes/{class_name}")
                    if res.status_code == 200:
                        st.warning(f"Class '{class_name}' deleted.")
                        time.sleep(0.5)
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete class: {e}")
                    
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# COLUMN 2: TRAINING CONTROLLER
# ==============================================================================
with col2:
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown('<div class="card-header-title">⚙️ 2. Model Training</div>', unsafe_allow_html=True)
    
    st.markdown("##### Hyperparameters")
    epochs = st.slider("Epochs", min_value=5, max_value=200, value=40, step=5)
    lr = st.selectbox("Learning Rate", options=[0.0001, 0.001, 0.01, 0.1], index=1)
    batch_size = st.selectbox("Batch Size", options=[4, 8, 16, 32, 64], index=2)
    
    # Validation checks before enabling training
    has_enough_classes = len(classes) >= 2
    has_enough_samples = all(sample_counts.get(c, 0) > 0 for c in classes) if classes else False
    
    can_train = has_enough_classes and has_enough_samples
    
    st.markdown("---")
    
    # Fetch status initially
    try:
        status_res = requests.get(f"{BACKEND_URL}/train/status").json()
        current_status = status_res.get("status", "idle")
    except Exception:
        current_status = "idle"
        status_res = {}
        
    # Trigger training
    if current_status == "training":
        st.markdown('<span class="status-badge status-training">⚡ Training in Progress</span>', unsafe_allow_html=True)
        st.session_state.training_active = True
    else:
        # Determine button state and tooltip
        btn_label = "Train Model"
        if not has_enough_classes:
            st.warning("⚠️ Need at least 2 classes configured to train.")
            st.button(btn_label, disabled=True, key="train_disabled_class")
        elif not has_enough_samples:
            st.warning("⚠️ All classes must have at least 1 image sample.")
            st.button(btn_label, disabled=True, key="train_disabled_samples")
        else:
            if st.button("🚀 Train Model", type="primary", use_container_width=True):
                try:
                    res = requests.post(
                        f"{BACKEND_URL}/train",
                        json={"epochs": epochs, "lr": lr, "batch_size": batch_size}
                    )
                    if res.status_code == 200:
                        st.session_state.training_active = True
                        st.rerun()
                    else:
                        st.error(res.json().get("detail", "Failed to start training"))
                except Exception as e:
                    st.error(f"Error: {e}")

    # Polling UI if training is in progress
    if st.session_state.training_active:
        progress_bar = st.progress(0.0)
        epoch_text = st.empty()
        loss_text = st.empty()
        acc_text = st.empty()
        
        # Poll server status
        while True:
            try:
                res = requests.get(f"{BACKEND_URL}/train/status").json()
                status = res.get("status", "idle")
                progress = res.get("progress", 0.0)
                cur_epoch = res.get("current_epoch", 0)
                tot_epochs = res.get("total_epochs", 0)
                metrics = res.get("metrics", {})
                
                # Update progress bar and texts
                progress_bar.progress(progress)
                epoch_text.markdown(f"**Epoch:** `{cur_epoch} / {tot_epochs}`")
                
                losses = metrics.get("loss", [])
                accuracies = metrics.get("accuracy", [])
                
                if losses:
                    loss_text.markdown(f"**Current Loss:** `{losses[-1]:.4f}`")
                if accuracies:
                    acc_text.markdown(f"**Current Accuracy:** `{accuracies[-1] * 100:.1f}%`")
                    
                if status == "completed":
                    st.success("🎉 Model training completed successfully!")
                    st.session_state.training_active = False
                    time.sleep(1.0)
                    st.rerun()
                    break
                elif status == "failed":
                    st.error(f"❌ Training failed: {res.get('error', 'Unknown error')}")
                    st.session_state.training_active = False
                    break
            except Exception as e:
                st.error(f"Failed to poll status: {e}")
                st.session_state.training_active = False
                break
                
            time.sleep(0.5)
            
    # Draw metrics charts if available
    metrics = status_res.get("metrics", {})
    losses = metrics.get("loss", [])
    accuracies = metrics.get("accuracy", [])
    
    if losses and accuracies:
        st.markdown("---")
        st.markdown("##### Training Curves")
        
        # Format data for Altair chart
        epoch_idx = list(range(1, len(losses) + 1))
        chart_data = pd.DataFrame({
            "Epoch": epoch_idx * 2,
            "Value": losses + accuracies,
            "Metric": ["Loss"] * len(losses) + ["Accuracy"] * len(accuracies)
        })
        
        # Plotting Altair Chart
        base = alt.Chart(chart_data).encode(
            x=alt.X("Epoch:Q", title="Epoch", axis=alt.Axis(tickMinStep=1)),
            color=alt.Color("Metric:N", scale=alt.Scale(domain=["Loss", "Accuracy"], range=["#ef4444", "#10b981"]))
        )
        
        line = base.mark_line(strokeWidth=3).encode(
            y=alt.Y("Value:Q", title="Value")
        )
        
        point = base.mark_point(size=40).encode(
            y="Value:Q",
            tooltip=["Epoch", "Metric", "Value"]
        )
        
        chart = (line + point).properties(
            width="container",
            height=200
        ).configure_view(
            strokeWidth=0
        ).configure_axis(
            gridColor="rgba(255, 255, 255, 0.05)",
            labelColor="#8b8a9f",
            titleColor="#ffffff"
        )
        
        st.altair_chart(chart, use_container_width=True)
    else:
        st.caption("Training metrics and curves will be visualized here once training is completed.")
        
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# COLUMN 3: PREVIEW & INFERENCE
# ==============================================================================
with col3:
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown('<div class="card-header-title">👁️ 3. Live Preview & Deploy</div>', unsafe_allow_html=True)
    
    # Test section requires model to be loaded/trained
    model_trained = health_data.get("model_loaded", False)
    
    if not model_trained:
        st.info("Awaiting model training. Configure classes and train the model in steps 1 & 2 to test classification predictions.")
    else:
        # Inference mode selection
        test_source = st.radio("Select test image source:", options=["📷 Webcam Preview", "🖼️ Upload Test Image"], horizontal=True)
        
        test_image = None
        if test_source == "📷 Webcam Preview":
            test_image = st.camera_input("Take a photo to classify:", key="test_camera_input")
        else:
            test_image = st.file_uploader("Upload an image to classify:", type=["jpg", "png", "jpeg"], key="test_file_input")
            
        if test_image:
            # Send to prediction endpoint
            img_bytes = test_image.getvalue()
            # Show original image in case of file uploader
            if test_source != "📷 Webcam Preview":
                st.image(test_image, width=220, caption="Uploaded Test Image")
                
            with st.spinner("Classifying image..."):
                try:
                    res = requests.post(
                        f"{BACKEND_URL}/predict",
                        files={"file": (getattr(test_image, "name", "test.jpg"), img_bytes, getattr(test_image, "type", "image/jpeg"))}
                    )
                    
                    if res.status_code == 200:
                        predictions_data = res.json()
                        predictions = predictions_data.get("predictions", {})
                        top_class = predictions_data.get("top_class", "")
                        
                        st.markdown("##### Predictions")
                        for class_name, confidence in predictions.items():
                            confidence_pct = confidence * 100
                            is_top = (class_name == top_class)
                            text_style = "font-weight: 800; color: #ffffff;" if is_top else "color: #8b8a9f;"
                            bar_fill_color = "linear-gradient(90deg, #10b981, #34d399);" if is_top else "linear-gradient(90deg, #6e44ff, #b249f8);"
                            
                            st.markdown(f"""
                            <div class="confidence-item" style="{'border-color: rgba(16, 185, 129, 0.4); box-shadow: 0 4px 12px 0 rgba(16, 185, 129, 0.1);' if is_top else ''}">
                                <div class="confidence-meta">
                                    <span style="{text_style}">{class_name} {"⭐" if is_top else ""}</span>
                                    <span style="{text_style}">{confidence_pct:.1f}%</span>
                                </div>
                                <div class="progress-track">
                                    <div class="progress-fill" style="width: {confidence_pct}%; background: {bar_fill_color};"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.error(res.json().get("detail", "Failed to run prediction"))
                except Exception as e:
                    st.error(f"Inference error: {e}")
                    
    # Persistence & Exports
    st.markdown("---")
    st.markdown("##### Model Export & Persistence")
    
    export_col, import_col = st.columns(2)
    
    with export_col:
        # Download ZIP button
        if not model_trained:
            st.button("📥 Download Model", disabled=True, key="download_disabled", use_container_width=True)
        else:
            try:
                # Trigger a GET request to export zip file
                res = requests.get(f"{BACKEND_URL}/export")
                if res.status_code == 200:
                    st.download_button(
                        label="📥 Download Model",
                        data=res.content,
                        file_name="neural_studio_model.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                else:
                    st.button("📥 Download Model", disabled=True, key="download_error", use_container_width=True)
            except Exception:
                st.button("📥 Download Model", disabled=True, key="download_conn_error", use_container_width=True)

    with import_col:
        # Load ZIP file
        imported_zip = st.file_uploader(
            "Upload Model ZIP:",
            type=["zip"],
            label_visibility="collapsed",
            key="import_zip_uploader"
        )
        if imported_zip:
            if st.button("📤 Load Model", type="secondary", use_container_width=True):
                with st.spinner("Restoring state..."):
                    try:
                        res = requests.post(
                            f"{BACKEND_URL}/import",
                            files={"file": (imported_zip.name, imported_zip.getvalue(), imported_zip.type)}
                        )
                        if res.status_code == 200:
                            st.success("Model loaded successfully!")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(res.json().get("detail", "Failed to load model"))
                    except Exception as e:
                        st.error(f"Import error: {e}")

    # Wipe System State
    st.markdown("---")
    if st.button("🚨 Reset Neural Studio", type="secondary", use_container_width=True):
        with st.spinner("Resetting backend..."):
            try:
                res = requests.post(f"{BACKEND_URL}/reset")
                if res.status_code == 200:
                    st.success("Session state cleared successfully.")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(res.json().get("detail", "Reset failed"))
            except Exception as e:
                st.error(f"Reset failed: {e}")
                
    st.markdown('</div>', unsafe_allow_html=True)
