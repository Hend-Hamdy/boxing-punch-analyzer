import streamlit as st
import tempfile
import os
import gdown
import cv2
import pandas as pd
from ultralytics import YOLO

# ——— Function to draw stats panel on each frame ———
def draw_stats_panel(frame, records, W):
    pw, ph = int(W * 0.8), 155
    px, py = (W - pw) // 2, 0
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 128, 0), thickness=cv2.FILLED)
    cols = ['Time', 'Side', 'Type', 'Target', 'Status', 'Quality']
    col_x = [px + 10 + i * 80 for i in range(len(cols))]
    y_h = py + 30
    for i, title in enumerate(cols):
        cv2.putText(frame, title, (col_x[i], y_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y0 = y_h + 30
    for rec in records[-5:]:
        for i, key in enumerate(cols):
            cv2.putText(frame, rec[key], (col_x[i], y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y0 += 25

# ——— Download best.pt from Google Drive on first run ———
MODEL_ID   = "1002vaPrGmRQ09nSc02yimtYCP_U6YOur"
MODEL_URL  = f"https://drive.google.com/uc?id={MODEL_ID}"
MODEL_PATH = "best.pt"
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ——— Load the YOLO model locally ———
model = YOLO('weights/best.pt')

# === Streamlit page setup ===
st.set_page_config(page_title="Boxing Punch Analyzer", layout="wide")
st.title("🥊 Boxing Punch Analyzer")
st.write(
    "Upload a boxing video (max 5 minutes) to detect punches, draw boxes, and display punch statistics."
)

# === Sidebar: optional email to save results ===
with st.sidebar:
    st.header("Save Your Run (optional)")
    user_email = st.text_input("Your email to save results")
    if user_email:
        st.info(f"Results will be saved for: {user_email}")

# === File uploader OR Google Drive link uploader ===
col1, col2 = st.columns(2)
video_path = None

with col1:
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
        tmp.write(uploaded_file.read())
        tmp.flush()
        video_path = tmp.name

with col2:
    drive_link = st.text_input("Or paste Google Drive share link")
    if drive_link and not uploaded_file:
        try:
            file_id = drive_link.split("/d/")[1].split("/")[0]
            dest = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            gdown.download(f"https://drive.google.com/uc?id={file_id}", dest, quiet=True)
            video_path = dest
            st.success("✅ Video downloaded from Google Drive")
        except Exception:
            st.error("❌ Could not download from that Drive link.")

# === Validate video length ===
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

# === Process button ===
if ready and st.button("▶️ Process Video"):
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    writer = cv2.VideoWriter(out_tmp.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    records = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        names = results.names

        for det in results.boxes:
            cls_id = int(det.cls)
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())

            # Draw bounding box
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

            if cls_id not in (0, 1, 2, 3):
                cv2.putText(frame, names[cls_id], (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

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
                parts = names[cls_id].split("_")
                side = parts[0]
                type_ = parts[1]
                target = parts[2]
                status_txt = "Landed" if stars > 0 else "Missed"

                secs = frame_idx / fps
                m2, s2 = divmod(int(secs), 60)
                time_str = f"{m2:02d}:{s2:02d}"

                records.append({
                    "Time":    time_str,
                    "Side":    side,
                    "Type":    type_,
                    "Target":  target,
                    "Status":  status_txt,
                    "Quality": quality
                })

        draw_stats_panel(frame, records, W)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    # ننسخ الفيديو المحفوظ إلى ملف جديد مؤقت عشان نقدر نعرضه
    with open(out_tmp.name, "rb") as f:
        video_bytes = f.read()

    display_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    with open(display_video.name, "wb") as f:
        f.write(video_bytes)

    # Show processed video with in-video stats panel
    st.video(display_video.name)

    # Show table of all records below
    if records:
        df = pd.DataFrame(records)
        df = df[["Time", "Side", "Type", "Target", "Status", "Quality"]]
        st.subheader("🥊 Punch Statistics")
        st.dataframe(df, height=300)

    st.download_button(
        "⬇️ Download Processed Video",
        data=open(display_video.name, "rb").read(),
        file_name="processed_boxing.mp4",
        mime="video/mp4"
    )

    if user_email:
        df.to_csv("saved_results.csv", mode="a", index=False,
                  header=not os.path.exists("saved_results.csv"))
        st.success(f"Results saved for {user_email}.")
