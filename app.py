import streamlit as st
import tempfile
import os
import cv2
import pandas as pd
from ultralytics import YOLO

st.set_page_config(page_title="Boxing Punch Analyzer", layout="wide")

# Header
st.title("🥊 Boxing Punch Analyzer")
st.write(
    "Upload a boxing video (max 5 minutes) to detect punches, draw boxes, "
    "and display punch statistics."
)

# Sidebar: optional save via email
with st.sidebar:
    st.header("Save Your Run (optional)")
    user_email = st.text_input("Your email to save results")
    if user_email:
        st.info(f"Results will be saved for: {user_email}")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a video file", type=["mp4", "avi", "mov"]
)
video_path = None
if uploaded_file:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
    tmp.write(uploaded_file.read())
    tmp.flush()
    video_path = tmp.name

# Validate video length
ready = False
if video_path:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    duration = frame_count / fps
    m, s = divmod(int(duration), 60)
    if duration > 300:
        st.error(f"❌ Video is too long: {m:02d}:{s:02d}. Maximum is 5 minutes.")
    else:
        st.success(f"✅ Video length: {m:02d}:{s:02d} (under 5 minutes)")
        ready = True

# Process button
if ready and st.button("▶️ Process Video"):
    progress = st.progress(0)
    status = st.empty()

    # Load model
    status.text("Loading YOLOv8 model…")
    model = YOLO('/content/drive/MyDrive/boxing_checkpoints/final_models/best.pt')
    progress.progress(10)

    # Prepare video writer
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    writer = cv2.VideoWriter(
        out_tmp.name,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (W, H)
    )
    progress.progress(20)

    # Collect stats
    records = []
    frame_idx = 0
    status.text("Detecting punches and collecting stats…")

    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        names = results.names

        for det in results.boxes:
            cls_id = int(det.cls)
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())

            # Choose box color
            if cls_id == 12:
                color = (0, 0, 255)
            elif cls_id == 13:
                color = (255, 0, 0)
            elif cls_id in (0, 1):
                color = (255, 255, 0)
            elif cls_id in (2, 3):
                color = (0, 255, 255)
            else:
                color = (255, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label text for classes != 0-3
            if cls_id not in (0, 1, 2, 3):
                cv2.putText(
                    frame, names[cls_id], (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
                )

            # Stats for punch classes
            if cls_id >= 4 and cls_id not in (12, 13):
                conf = float(det.conf[0]) if hasattr(det.conf, "__len__") else float(det.conf)
                if conf > 0.85:
                    stars = 3
                elif conf > 0.65:
                    stars = 2
                elif conf > 0.55:
                    stars = 1
                else:
                    stars = 0
                quality = " ".join("*" for _ in range(stars))
                label = names[cls_id]
                parts = label.split("_")
                side = parts[0]
                type_ = parts[1]
                target = parts[2]
                status_txt = "Landed" if stars > 0 else "Missed"
                secs = frame_idx / fps
                m2, s2 = divmod(int(secs), 60)
                time_str = f"{m2:02d}:{s2:02d}"
                records.append({
                    "Time": time_str,
                    "Side": side,
                    "Type": type_,
                    "Target": target,
                    "Status": status_txt,
                    "Quality": quality
                })

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    progress.progress(80)
    status.text("Processing complete!")

    # Show output video
    st.video(out_tmp.name)
    if records:
        df = pd.DataFrame(records)
        st.subheader("Punch Statistics")
        st.dataframe(df, height=300)

    st.download_button(
        "⬇️ Download Processed Video",
        data=open(out_tmp.name, "rb").read(),
        file_name="processed_boxing.mp4",
        mime="video/mp4"
    )

    if user_email:
        save_path = "saved_results.csv"
        df.to_csv(save_path, index=False, mode="a", header=not os.path.exists(save_path))
        st.success(f"Results saved for {user_email}.")
