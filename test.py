from flask import Flask, render_template, Response, jsonify, request
import face_recognition
import cv2
import os
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Global variables
recognized_name = None
marked_today = False
video_capture = None  # Initialize the webcam variable

# Folder containing known face images
known_faces_folder = "known_faces"

# Lists to hold known face encodings and names
known_face_encodings = []
known_face_names = []

# Load known faces
def load_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings.clear()
    known_face_names.clear()
    for image_file in os.listdir(known_faces_folder):
        if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            image_path = os.path.join(known_faces_folder, image_file)
            known_image = face_recognition.load_image_file(image_path)
            known_face_encodings.append(face_recognition.face_encodings(known_image)[0])
            known_face_names.append(os.path.splitext(image_file)[0])

load_known_faces()

# Define the path for the attendance file
attendance_file = "Attendance.xlsx"

# Create the attendance DataFrame if the file does not exist
if not os.path.exists(attendance_file):
    df_columns = ["ID", "Name", "Date", "Time", "Day", "Year", "Status"]
    pd.DataFrame(columns=df_columns).to_excel(attendance_file, index=False)

def mark_attendance(id, name):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    day_str = now.strftime("%A")
    year_str = now.strftime("%Y")
    status = "Present"

    existing_df = pd.read_excel(attendance_file, engine='openpyxl')
    if not ((existing_df['Name'] == name) & (existing_df['Date'] == date_str)).any():
        new_entry = pd.DataFrame([[id, name, date_str, time_str, day_str, year_str, status]], columns=existing_df.columns)
        with pd.ExcelWriter(attendance_file, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            new_entry.to_excel(writer, index=False, header=False, startrow=len(existing_df) + 1)
        return True
    return False

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/take_attendance')
def take_attendance():
    global video_capture
    if video_capture is None or not video_capture.isOpened():
        video_capture = cv2.VideoCapture(0)  # Re-initialize the webcam
    return render_template('take_attendance.html')

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    global recognized_name, marked_today
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                id = first_match_index + 1
                attendance_marked = mark_attendance(id, name)

                recognized_name = name
                marked_today = attendance_marked
                message = f"{name} marked present" if attendance_marked else f"{name} already marked today"
            else:
                message = "Unknown face detected"

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, message, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/close_webcam')
def close_webcam_route():
    global video_capture, recognized_name, marked_today
    close_webcam()
    message = f"Attendance marked for {recognized_name}" if recognized_name else "No attendance marked"
    return jsonify({"message": message})

def close_webcam():
    global video_capture
    if video_capture is not None and video_capture.isOpened():
        video_capture.release()
    cv2.destroyAllWindows() 

@app.route('/add_user', methods=['POST'])
def add_user():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    name = request.form['name']

    if file.filename == '':
        return "No selected file", 400

    # Save the uploaded image
    image_path = os.path.join(known_faces_folder, name + '.jpg')
    file.save(image_path)

    # Load the new user's face encoding
    new_user_image = face_recognition.load_image_file(image_path)
    new_user_encoding = face_recognition.face_encodings(new_user_image)[0]

    # Save the new user encoding and name
    known_face_encodings.append(new_user_encoding)
    known_face_names.append(name)

    # Generate user ID
    user_id = len(known_face_names)  # Assuming IDs are based on the list length

    # Prepare image URL to pass to the template
    image_url = os.path.join('known_faces', name + '.jpg')

    # Render the success template
    return render_template('registration_success.html', id=user_id, name=name, image_url=image_url)

@app.route('/attendance_details')
def attendance_details():
    if os.path.exists(attendance_file):
        df = pd.read_excel(attendance_file)
        data = df.to_dict(orient='records')  # Convert DataFrame to list of dictionaries
        return render_template('attendance_details.html', data=data)
    else:
        return "No attendance records found."

if __name__ == '__main__':
    app.run(debug=True)
