import cv2 as cv

capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    haar_cascade = cv.CascadeClassifier('haar_face.xml')

    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    for (x, y, w, h) in faces_rect:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
        cv.putText(frame, 'Face', (x, y - 6), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 2)

    cv.imshow('Face Detector', frame)

    if cv.waitKey(20) & 0xFF == ord('x'):
        break

print(f"Number of faces detected: {len(faces_rect)}")

capture.release()
cv.destroyAllWindows()
