import cv2
import google.generativeai as genai
import time

# Set your API key securely
GOOGLE_API_KEY = 'XXXXXXX'
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize video capture from the webcam
cap = cv2.VideoCapture(0)

# Check if the video capture is opened
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Load YOLO object detection model (you'll need the config and weights files)
# Make sure you have yolov3.weights and yolov3.cfg in the same directory
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame was successfully read
    if not ret:
        print("Error: Failed to capture image")
        break

    # Get height, width, and channels of the frame
    height, width, channels = frame.shape

    # Prepare the frame for the YOLO model
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists for detected objects
    class_ids = []
    confidences = []
    boxes = []

    # Loop over each of the detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = int(scores.argmax())
            confidence = scores[class_id]
            if confidence > 0.5:  # Filter out weak detections
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Get the coordinates for the rectangle
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression to remove duplicates
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(class_ids[i])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Extract the detected object
            detected_object = frame[y:y+h, x:x+w]
            cv2.imwrite('detected_object.jpg', detected_object)
            
            # Upload the detected object image and interact with the Generative AI service
            sample_file = genai.upload_file(path='detected_object.jpg', display_name="Detected Object")
            print(f"Uploaded file '{sample_file.display_name}' as: {sample_file.uri}")

            # Choose a Gemini model
            model = genai.GenerativeModel(model_name="gemini-1.5-pro")

            # Prompt the model with text and the previously uploaded image
            response = model.generate_content([sample_file, "Tell me more about this object"])
            
            # Display the response
            print(response.text)
            
            # To avoid spamming the API, add a short delay
            time.sleep(5)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
