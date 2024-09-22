from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
import face_recognition
import cv2
import os
import pandas as pd
import datetime

app = Flask(__name__)

# Folder for known faces
known_faces_folder = "known_faces"
attendance_file = "Attendance.xlsx"  # Excel file for attendance

# Lists for known face encodings and names
known_face_encodings = []
known_face_names = []

# Supported image formats
supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

# Load known faces
def load_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    for image_file in os.listdir(known_faces_folder):
        if image_file.lower().endswith(supported_formats):
            image_path = os.path.join(known_faces_folder, image_file)
            known_image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(known_image)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(os.path.splitext(image_file)[0])
    print(f"Loaded {len(known_face_encodings)} known faces.")

# Load known faces at startup
load_known_faces()

# Function to close webcam
def close_webcam():
    global video_capture
    if video_capture.isOpened():
        video_capture.release()
        cv2.destroyAllWindows()

# Webcam feed
video_capture = cv2.VideoCapture(0)

# Function to log attendance
def log_attendance(name):
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")

    # Load the attendance Excel file or create a new DataFrame
    if not os.path.exists(attendance_file):
        df = pd.DataFrame(columns=["ID", "Name", "Date", "Time", "Day", "Year", "Status"])
    else:
        df = pd.read_excel(attendance_file)

    # Check if the person is already marked as present for the day
    if ((df["Name"] == name) & (df["Date"] == date_str)).any():
        return False  # Already marked, return False to indicate attendance is already marked

    time_str = now.strftime("%H:%M:%S")
    day_str = now.strftime("%A")
    year_str = now.strftime("%Y")

    # Add new entry
    new_entry = {
        "ID": len(df) + 1,
        "Name": name,
        "Date": date_str,
        "Time": time_str,
        "Day": day_str,
        "Year": year_str,
        "Status": "Present"
    }
    df = df.append(new_entry, ignore_index=True)
    df.to_excel(attendance_file, index=False)

    return True  # Newly marked, return True to indicate new attendance marked

# Recognition function
def generate_frames():
    attendance_marked = False
    recognized_name = ""  # Track recognized person's name
    marked_today = None  # Track if the person is already marked for today

    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                # Log attendance for recognized faces
                if name != "Unknown" and not attendance_marked:
                    marked_today = log_attendance(name)
                    attendance_marked = True
                    recognized_name = name
                    break  # Stop further processing after recognition

                # Draw a rectangle around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Encode the frame and yield as bytes for the stream
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Close the webcam once attendance is marked
    if attendance_marked:
        close_webcam()


# Update the /close_webcam route to handle attendance status
@app.route('/close_webcam')
def close_webcam_route():
    global recognized_name, marked_today
    if recognized_name:
        message = f"Attendance marked for {recognized_name}" if marked_today else f"{recognized_name} already marked today"
        return jsonify({"message": message})
    return jsonify({"message": "No attendance marked"})

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/take_attendance')
def take_attendance():
    return render_template('take_attendance.html')

@app.route('/attendance_details')
def attendance_details():
    if os.path.exists(attendance_file):
        df = pd.read_excel(attendance_file)
        data = df.to_dict(orient='records')  # Convert DataFrame to list of dictionaries
        return render_template('attendance_details.html', data=data)
    else:
        return "No attendance records found."

@app.route('/add_user', methods=['POST'])
def add_user():
    if 'file' not in request.files or 'name' not in request.form:
        return "No file or name provided", 400

    file = request.files['file']
    name = request.form['name']

    if file.filename == '' or not file.filename.lower().endswith(supported_formats):
        return "Invalid file format", 400

    filename = f"{name}.jpg"
    file_path = os.path.join(known_faces_folder, filename)
    file.save(file_path)

    # Reload known faces
    load_known_faces()

    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
