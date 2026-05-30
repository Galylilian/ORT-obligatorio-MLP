import streamlit as st
import requests
from PIL import Image
from io import BytesIO

API_URL = "http://localhost:8080"

st.title("Detector de paciente en camilla")

file = st.file_uploader("Subir imagen", type=["jpg", "png"])

if file:
    # =========================
    # Mostrar imagen
    # =========================
    image = Image.open(file)
    st.image(image, caption="Imagen original", width="stretch")

    # =========================
    # CNN
    # =========================
    file.seek(0)
    cnn_resp = requests.post(
        f"{API_URL}/predict",
        files={"file": file}
    )
    cnn_result = cnn_resp.json()

    # =========================
    # YOLO
    # =========================
    file.seek(0)
    yolo_resp = requests.post(
        f"{API_URL}/predict_yolo",
        files={"file": file}
    )
    yolo_result = yolo_resp.json()

    # =========================
    # RESULTADOS
    # =========================
    st.subheader("Resultados")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### CNN")
        st.json(cnn_result)

    with col2:
        st.write("### YOLO")
        st.json(yolo_result)

    # =========================
    # GRAD-CAM
    # =========================
    file.seek(0)
    grad_resp = requests.post(
        f"{API_URL}/gradcam",
        files={"file": file}
    )

    # ✅ VALIDACIÓN CORRECTA
    content_type = grad_resp.headers.get("content-type", "")

    if grad_resp.status_code == 200 and "image" in content_type:
        try:
            grad_img = Image.open(BytesIO(grad_resp.content))
            st.image(grad_img, caption="Grad-CAM", width="stretch")
        except Exception as e:
            st.error("Error al interpretar GradCAM")
            st.text(str(e))

    else:
        st.warning("No se pudo generar Grad-CAM")

        # 🔥 MOSTRAR ERROR REAL (clave para debug)
        try:
            st.text(grad_resp.text)
        except:
            st.text("Respuesta inválida del servidor")