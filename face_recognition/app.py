# Imports for face recognition app
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import cv2
import os

net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Initialize file paths for attendance and student data
attendance_file = "attendance.csv"
student_data_file = "student_data.csv"

# Initialize CSV files if not present
if not os.path.exists(attendance_file):
    pd.DataFrame(columns=["Name", "ID", "Timestamp"]).to_csv(attendance_file, index=False)

if not os.path.exists(student_data_file):
    pd.DataFrame(columns=["Name", "ID", "Face_Encoding"]).to_csv(student_data_file, index=False)

# Helper function to load student data
def load_student_data():
    return pd.read_csv(student_data_file)

# Helper function to detect faces using OpenCV DNN module
def detect_faces(image):
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]
            faces.append(face)
    return faces

# Helper function to save student data
def save_student_data(name, student_id, face_encoding):
    student_data = load_student_data()
    new_data = {"Name": name, "ID": student_id, "Face_Encoding": face_encoding}
    student_data = student_data.append(new_data, ignore_index=True)
    student_data.to_csv(student_data_file, index=False)

# Helper function to mark attendance
def mark_attendance(name, student_id):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    attendance_data = pd.read_csv(attendance_file)
    new_entry = {"Name": name, "ID": student_id, "Timestamp": current_time}
    attendance_data = attendance_data.append(new_entry, ignore_index=True)
    attendance_data.to_csv(attendance_file, index=False)

# Streamlit UI
def run_app():
    st.title("Face Recognition Attendance System")

    # Sidebar for navigation
    option = st.sidebar.selectbox("Choose an option", ["Home", "Register", "Mark Attendance", "View Attendance"])

    # Home Page
    if option == "Home":
        st.write("Welcome to the Face Recognition Attendance System.")

    # Registration Page
    elif option == "Register":
        st.subheader("Student Registration")
        student_name = st.text_input("Enter Student Name")
        student_id = st.text_input("Enter Student ID")

        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            faces = detect_faces(image)
            if faces:
                face_encoding = faces[0].tolist()  # Simplified encoding
                save_student_data(student_name, student_id, face_encoding)
                st.success(f"Student {student_name} (ID: {student_id}) registered successfully!")
            else:
                st.error("No face detected in the image.")

    # Mark Attendance Page
    elif option == "Mark Attendance":
        st.subheader("Mark Attendance")
        uploaded_file = st.file_uploader("Upload Image for Attendance", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            faces = detect_faces(image)
            if faces:
                face_encoding = faces[0].tolist()  # Simplified encoding
                student_data = load_student_data()
                match = False
                for index, row in student_data.iterrows():
                    known_encoding = eval(row["Face_Encoding"])
                    distance = np.linalg.norm(np.array(known_encoding) - np.array(face_encoding))
                    if distance < 0.5:  # Threshold for matching
                        mark_attendance(row["Name"], row["ID"])
                        st.success(f"Attendance marked for {row['Name']} (ID: {row['ID']}).")
                        match = True
                        break
                if not match:
                    st.error("No match found. Please register first.")

    # View Attendance Page
    elif option == "View Attendance":
        st.subheader("Attendance Records")
        attendance_data = pd.read_csv(attendance_file)
        st.dataframe(attendance_data)
        if st.button("Download Attendance"):
            st.download_button(label="Download Attendance", data=open(attendance_file).read(), file_name="attendance.csv")

if __name__ == "__main__":
    run_app()

