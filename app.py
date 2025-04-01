import streamlit as st
from PIL import Image
import pandas as pd
import sqlite3
from datetime import datetime
from smart_detect import smart_analyze
from vlm import analyze_with_vlm
import requests
import base64
import io

# DB 연결
conn = sqlite3.connect("logs.db", check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    image_name TEXT,
    summary TEXT,
    direction TEXT,
    hazard_level TEXT,
    signal_color TEXT,
    latitude REAL,
    longitude REAL
)
""")
conn.commit()

def save_log(name, result, lat, lon):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("""
        INSERT INTO logs (timestamp, image_name, summary, direction, hazard_level, signal_color, latitude, longitude)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now, name, result['summary'], result['direction'],
        result['hazard_level'], result['signal_color'], lat, lon
    ))
    conn.commit()

# Roboflow YOLO-World inference API
ROBOFLOW_API_KEY = "Your API key"
YOLO_WORLD_ENDPOINT = "https://infer.roboflow.com/foundation/yolo_world"

# 객체 설명 텍스트 (Open Vocabulary)
open_vocab = ["rock", "rail", "rail damage", "vehicle", "signal", "worker", "tree", "obstacle", "barrier"]

def run_yolo_world(image: Image.Image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    response = requests.post(
        YOLO_WORLD_ENDPOINT,
        params={"api_key": ROBOFLOW_API_KEY},
        json={"image": img_str, "classes": open_vocab}
    )

    preds = response.json()
    results = preds.get("predictions", [])
    unknowns = [p for p in results if p['class'] not in open_vocab]

    return results, unknowns

# Streamlit UI
st.set_page_config(page_title="Railway Safety AI", layout="centered")
st.title("🚦 Railway Vision AI (YOLO-World + VLM Edition)")

menu = st.sidebar.selectbox("Menu", ["Analyze Image", "View Logs"])

if menu == "Analyze Image":
    uploaded_file = st.file_uploader("Upload railway image", type=["jpg", "jpeg", "png"])
    lat = st.number_input("Latitude (optional)", value=0.0, format="%.6f")
    lon = st.number_input("Longitude (optional)", value=0.0, format="%.6f")

    mode = st.selectbox("🧠 분석 방식 선택", ["YOLO + VLM", "YOLO-World + VLM"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Analyzing image..."):
            if mode == "YOLO + VLM":
                result = smart_analyze(image)
            elif mode == "YOLO-World + VLM":
                yolo_results, unknowns = run_yolo_world(image)
                summary = f"YOLO-World Detected {len(yolo_results)} objects. Unknowns: {len(unknowns)}"

                if unknowns:
                    vlm_question = "What are the unknown objects in this image?"
                else:
                    vlm_question = None

                vlm_caption = analyze_with_vlm(image, vlm_question)
                full_summary = summary + "\nVLM says: " + vlm_caption

                result = {
                    "summary": full_summary,
                    "labels": [r['class'] for r in yolo_results],
                    "signal_color": "Unknown",
                    "hazard_level": "High" if len(unknowns) > 0 else "Low",
                    "direction": "Unknown"
                }

        st.success("Analysis Complete")
        st.markdown(f"**Summary:**\n{result['summary']}")
        st.markdown(f"**Signal Color:** `{result['signal_color']}`")
        st.markdown(f"**Hazard Level:** `{result['hazard_level']}`")
        st.markdown(f"**Direction:** `{result['direction']}`")

        if result['hazard_level'] == "High":
            st.error("🚨 High hazard detected!")
            st.audio("https://www.soundjay.com/buttons/sounds/beep-07.mp3")

        save_log(uploaded_file.name, result, lat, lon)
        st.info("Log saved successfully.")

elif menu == "View Logs":
    st.subheader("📋 Logs")
    c.execute("SELECT * FROM logs ORDER BY timestamp DESC")
    rows = c.fetchall()
    if rows:
        df = pd.DataFrame(rows, columns=["ID", "Timestamp", "Image", "Summary", "Direction", "Hazard", "Signal", "Lat", "Lon"])
        st.dataframe(df, use_container_width=True)

        st.subheader("📊 Hazard Distribution")
        st.bar_chart(df["Hazard"].value_counts())

        st.subheader("🗺️ Map View")
        if not df[["Lat", "Lon"]].dropna().empty:
            st.map(df[["Lat", "Lon"]].dropna())

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", csv, file_name="railway_logs.csv")
    else:
        st.info("No logs yet.")