import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import datetime
import time
from PIL import Image
import pyttsx3

# Constants
haarcasecade_path = "haarcascade_frontalface_default.xml"
trainimagelabel_path = "TrainingImageLabel/Trainner.yml"
trainimage_path = "TrainingImage"
studentdetail_path = "StudentDetails/studentdetails.csv"
attendance_path = "Attendance"

# Ensure required directories exist
for path in [trainimage_path, "TrainingImageLabel", "StudentDetails", attendance_path]:
    if not os.path.exists(path):
        os.makedirs(path)

def text_to_speech(user_text):
    engine = pyttsx3.init()
    engine.say(user_text)
    engine.runAndWait()

def take_attendance(subject):
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        try:
            recognizer.read(trainimagelabel_path)
        except:
            st.error("Model not found, please train the model first")
            text_to_speech("Model not found, please train the model first")
            return

        face_cascade = cv2.CascadeClassifier(haarcasecade_path)
        df = pd.read_csv(studentdetail_path)
        cam = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        col_names = ["Enrollment", "Name"]
        attendance = pd.DataFrame(columns=col_names)
        
        stframe = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        start_time = time.time()
        duration = 20  # seconds
        
        while True:
            ret, im = cam.read()
            if not ret:
                st.error("Failed to access camera")
                break
                
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5)
            
            for (x, y, w, h) in faces:
                Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
                if conf < 70:
                    aa = df.loc[df["Enrollment"] == Id]["Name"].values
                    tt = str(Id) + "-" + str(aa[0])
                    attendance.loc[len(attendance)] = [Id, aa[0]]
                    cv2.rectangle(im, (x, y), (x+w, y+h), (0, 260, 0), 4)
                    cv2.putText(im, str(tt), (x+h, y), font, 1, (255, 255, 0), 4)
                else:
                    Id = "Unknown"
                    tt = str(Id)
                    cv2.rectangle(im, (x, y), (x+w, y+h), (0, 25, 255), 7)
                    cv2.putText(im, str(tt), (x+h, y), font, 1, (0, 25, 255), 4)
            
            # Update progress
            elapsed_time = time.time() - start_time
            progress = min(elapsed_time / duration, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Time remaining: {int(duration - elapsed_time)} seconds")
            
            # Display the frame
            stframe.image(im, channels="BGR", use_column_width=True)
            
            if elapsed_time > duration:
                break
# ...existing code...
        while True:
            ret, im = cam.read()
            if not ret:
                st.error("Failed to access camera")
                break
                
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5)
            
            for (x, y, w, h) in faces:
                Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
                if conf < 70:
                    aa = df.loc[df["Enrollment"] == Id]["Name"].values
                    tt = str(Id) + "-" + str(aa[0])
                    attendance.loc[len(attendance)] = [Id, aa[0]]
                    cv2.rectangle(im, (x, y), (x+w, y+h), (0, 260, 0), 4)
                    cv2.putText(im, str(tt), (x+h, y), font, 1, (255, 255, 0), 4)
                else:
                    Id = "Unknown"
                    tt = str(Id)
                    cv2.rectangle(im, (x, y), (x+w, y+h), (0, 25, 255), 7)
                    cv2.putText(im, str(tt), (x+h, y), font, 1, (0, 25, 255), 4)
            
            # Update progress
            elapsed_time = time.time() - start_time
            progress = min(elapsed_time / duration, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Time remaining: {int(duration - elapsed_time)} seconds")
            
            # Display the frame
            stframe.image(im, channels="BGR", use_column_width=True)
            
            if elapsed_time > duration:
                break

            # Only create the stop button once, outside the loop
            if 'stop_attendance' not in st.session_state:
                st.session_state['stop_attendance'] = False
            if st.button("Stop", key="stop_attendance_main"):
                st.session_state['stop_attendance'] = True
            if st.session_state['stop_attendance']:
                break
# ...existing code...
            # Only create the stop button once, outside the loop
            if 'stop_attendance' not in st.session_state:
                st.session_state['stop_attendance'] = False
            if st.button("Stop", key="stop_attendance_main"):
                st.session_state['stop_attendance'] = True
            if st.session_state['stop_attendance']:
                break
        
        cam.release()
        cv2.destroyAllWindows()
        
        if not attendance.empty:
            attendance = attendance.drop_duplicates(["Enrollment"], keep="first")
            ts = time.time()
            date = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            timeStamp = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            Hour, Minute, Second = timeStamp.split(":")
            
            path = os.path.join(attendance_path, subject)
            if not os.path.exists(path):
                os.makedirs(path)
                
            fileName = f"{path}/{subject}_{date}_{Hour}-{Minute}-{Second}.csv"
            attendance[date] = 1
            attendance.to_csv(fileName, index=False)
            
            st.success(f"Attendance taken successfully for {subject}")
            text_to_speech(f"Attendance taken successfully for {subject}")
            
            # Display attendance
            st.dataframe(attendance)
        else:
            st.warning("No faces detected during the session")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def register_new_student():
    st.subheader("Register New Student")
    
    enrollment = st.text_input("Enrollment Number", key="enrollment")
    name = st.text_input("Name", key="name")
    
    if st.button("Start Registration"):
        if not enrollment or not name:
            st.error("Please enter both enrollment number and name")
            return
            
        if not enrollment.isdigit():
            st.error("Enrollment number must be numeric")
            return
            
        try:
            cam = cv2.VideoCapture(0)
            face_cascade = cv2.CascadeClassifier(haarcasecade_path)
            
            stframe = st.empty()
            sample_count = 0
            max_samples = 50
            
            while sample_count < max_samples:
                ret, frame = cam.read()
                if not ret:
                    st.error("Failed to access camera")
                    break
                    
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.2, 5)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 260, 0), 4)
                    sample_count += 1
                    cv2.imwrite(f"{trainimage_path}/User.{enrollment}.{sample_count}.jpg", 
                              gray[y:y+h, x:x+w])
                
                stframe.image(frame, channels="BGR", use_column_width=True)
                st.progress(sample_count / max_samples)
                
                if st.button("Stop Registration", key=f"stop_registration_{sample_count}"):
                    break
                    
            cam.release()
            cv2.destroyAllWindows()
            
            if sample_count > 0:
                # Update student details
                student_details = pd.DataFrame({
                    "Enrollment": [enrollment],
                    "Name": [name]
                })
                
                if os.path.exists(studentdetail_path):
                    existing_df = pd.read_csv(studentdetail_path)
                    student_details = pd.concat([existing_df, student_details], ignore_index=True)
                
                student_details.to_csv(studentdetail_path, index=False)
                st.success(f"Successfully registered {name} with {sample_count} samples")
                text_to_speech(f"Successfully registered {name}")
                
                # Train the model
                if st.button("Train Model", key="train_after_registration"):
                    train_model()
            else:
                st.error("No face detected during registration")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

def train_model():
    try:
        faces = []
        ids = []
        
        for path in os.listdir(trainimage_path):
            if path.endswith(".jpg"):
                image_path = os.path.join(trainimage_path, path)
                img = Image.open(image_path).convert('L')
                img_np = np.array(img, 'uint8')
                id = int(os.path.split(image_path)[-1].split(".")[1])
                faces.append(img_np)
                ids.append(id)
        
        if not faces:
            st.error("No training images found")
            return
            
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(ids))
        recognizer.save(trainimagelabel_path)
        
        st.success("Model trained successfully")
        text_to_speech("Model trained successfully")
        
    except Exception as e:
        st.error(f"An error occurred during training: {str(e)}")

def view_attendance():
    st.subheader("View Attendance Records")
    
    subjects = [d for d in os.listdir(attendance_path) 
               if os.path.isdir(os.path.join(attendance_path, d))]
    
    if not subjects:
        st.warning("No attendance records found")
        return
        
    selected_subject = st.selectbox("Select Subject", subjects)
    
    if selected_subject:
        subject_path = os.path.join(attendance_path, selected_subject)
        files = [f for f in os.listdir(subject_path) if f.endswith('.csv')]
        
        if not files:
            st.warning(f"No attendance records found for {selected_subject}")
            return
            
        selected_file = st.selectbox("Select Date", files)
        
        if selected_file:
            file_path = os.path.join(subject_path, selected_file)
            df = pd.read_csv(file_path)
            st.dataframe(df)

def main():
    st.set_page_config(
        page_title="Attendance Management System",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Attendance Management System")
    
    menu = ["Home", "Register New Student", "Take Attendance", "View Attendance", "Train Model"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.write("""
        ### Welcome to the Attendance Management System
        
        This system uses face recognition to manage student attendance.
        
        **Features:**
        - Register new students with face recognition
        - Take attendance using face recognition
        - View attendance records
        - Train the face recognition model
        
        Please select an option from the sidebar to get started.
        """)
        
    elif choice == "Register New Student":
        register_new_student()
        
    elif choice == "Take Attendance":
        st.subheader("Take Attendance")
        subject = st.text_input("Enter Subject Name")
        if st.button("Start Attendance", key="start_attendance"):
            if not subject:
                st.error("Please enter a subject name")
            else:
                take_attendance(subject)
                
    elif choice == "View Attendance":
        view_attendance()
        
    elif choice == "Train Model":
        st.subheader("Train Face Recognition Model")
        if st.button("Start Training", key="start_training"):
            train_model()

if __name__ == "__main__":
    main()