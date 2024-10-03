import cv2
from fer import FER

# Initialize the camera
cap = cv2.VideoCapture(1)  # Ensure the correct camera index

# Initialize the FER emotion detector
emotion_detector = FER()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    # Detect emotions in the frame
    emotions = emotion_detector.detect_emotions(frame)
    # If emotions are detected, display them
    if emotions:
        for emotion in emotions:
            (x, y, w, h) = emotion['box']
            dominant_emotion = max(emotion['emotions'], key=emotion['emotions'].get)
            confidence = emotion['emotions'][dominant_emotion]
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Display the dominant emotion with confidence
            cv2.putText(frame, f"{dominant_emotion} ({confidence:.2f})", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


