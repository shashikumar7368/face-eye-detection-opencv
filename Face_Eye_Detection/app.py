
import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.markdown("""
<style>
/* Main application background */
.stApp {
    background: linear-gradient(135deg, #f0fdf4, #ecfeff);
}

/* Main content width and spacing */
.block-container {
    max-width: 1000px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

/* Main heading */
h1 {
    color: #166534 !important;
    text-align: center;
    font-size: 2.6rem !important;
    font-weight: 800 !important;
}

/* Subheadings */
h2, h3 {
    color: #15803d !important;
}

/* Paragraph text */
p {
    color: #374151;
    font-size: 1.05rem;
}

/* Upload and camera containers */
[data-testid="stFileUploader"],
[data-testid="stCameraInput"] {
    background: rgba(255, 255, 255, 0.85);
    padding: 18px;
    border: 1px solid #bbf7d0;
    border-radius: 16px;
    box-shadow: 0 4px 14px rgba(22, 101, 52, 0.07);
}

/* Buttons */
.stButton > button {
    background: #16a34a;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background: #166534;
    color: white;
    border: none;
    transform: translateY(-2px);
}

/* Detection image */
[data-testid="stImage"] img {
    border-radius: 16px;
    border: 2px solid #bbf7d0;
    box-shadow: 0 6px 20px rgba(22, 101, 52, 0.12);
}

/* Success and information messages */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Footer */
.footer {
    text-align: center;
    color: #6b7280;
    font-size: 0.9rem;
    margin-top: 35px;
    padding: 15px;
    border-top: 1px solid #d1fae5;
}

/* Responsive heading on smaller screens */
@media (max-width: 600px) {
    h1 {
        font-size: 1.9rem !important;
    }

    .block-container {
        padding: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)


st.markdown(
    "<h1>Face & Eye Detection</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>"
    "An AI-powered computer vision application that detects "
    "human faces and eyes using Python and OpenCV."
    "</p>",
    unsafe_allow_html=True
)

st.write("")
st.divider()


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

if face_cascade.empty() or eye_cascade.empty():
    st.error("Could not load the Haar Cascade files.")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a photo",
    type=["jpg", "jpeg", "png"]
)

camera_image = st.camera_input("Or take a photo with your camera")

image_source = camera_image if camera_image is not None else uploaded_file

if image_source is not None:
    image = Image.open(image_source).convert("RGB")
    image_array = np.array(image)
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    eye_count = 0

    for (x, y, w, h) in faces:
        cv2.rectangle(
            image_array,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        face_gray = gray[y:y + h, x:x + w]
        face_color = image_array[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(10, 10)
        )

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                face_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (255, 0, 0),
                2
            )
            eye_count += 1

    st.subheader("Detection Result")

    st.image(
        image_array,
        caption="Green boxes: faces | Blue boxes: eyes",
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Faces Detected",
            value=len(faces),
            border=True
        )

    with col2:
        st.metric(
            label="Eyes Detected",
            value=eye_count,
            border=True
        )



    if len(faces) == 0:
        st.warning(
            "No faces were detected. Try a clear, front-facing photo."
        )

st.caption("Built with Python, OpenCV, Streamlit and Haar Cascades.")

st.markdown("""
<div class="footer">
    Built with Python • OpenCV • Streamlit • Haar Cascades
    <br>
    Face and Eye Detection Project
</div>
""", unsafe_allow_html=True)
