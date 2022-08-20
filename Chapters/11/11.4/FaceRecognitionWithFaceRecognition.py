# Detect the features of face
from PIL import Image, ImageDraw, ImageFont
import face_recognition
import cv2
import numpy as np

def DrawFacialFeatures(path):
    
    # Load the image into an array of NumPy
    image = face_recognition.load_image_file(path)
    # Find all the face features from image
    faceLandmarks = face_recognition.face_landmarks(image)
    
    print("{} faces observed".format(len(faceLandmarks)))
    
    # The key of face feature
    facialFeatures = ["chin", "left_eyebrow", "right_eyebrow", "nose_bridge", "nose_tip", "left_eye", "right_eye", "top_lip", "bottom_lip"]
      
    for landmark in faceLandmarks:
        
        # Convert the NumPy array into PIL.Image.Image object
        pilImage = Image.fromarray(image)
        # Convert the PIL.Image.Image object into PIL.ImageDraw.ImageDraw drawable object
        drawable = ImageDraw.Draw(pilImage)
        
        # According to the facial features
        for facialFeature in facialFeatures:
            
            # DRaw the line with width 5
            drawable.line(landmark[facialFeature], width = 5)
            
        pilImage.show()

def LoadFacialImage(path):
    
    # Load the image object into memory with array
    image = face_recognition.load_image_file(path)
    # Encode the image into 128 dimensions facial matrix, and return by NumPy array
    encodings = face_recognition.face_encodings(image)
    
    return encodings
    
def DrawChinese(image, text, position, textColor = (0, 255, 0), textSize = 30):

    if (isinstance(image, np.ndarray)):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    draw = ImageDraw.Draw(image)
    fontStyle = ImageFont.truetype(
        "Songti.ttc", textSize, encoding = "utf-8")
    
    draw.text(position, text, textColor, font = fontStyle)
    
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    
def LiveRecognize(paths):
    
    # https://medium.com/analytics-vidhya/face-recognition-using-face-recognition-api-e7fa4dcabbc3
    # Grab the object of realtime webcab object
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        
        print("Cannot open camera")
        exit()
    
    # Create known facial encoding array
    knownFacialEncodings = []
    
    for path in paths:
    
        # Load the known facial object
        facialEncodings = LoadFacialImage(path)
        knownFacialEncodings.append(facialEncodings[0])
            
    # Create the name of the known facial object
    knownFaceNames = ["Personnel 1", "Personnel 2"]
    
    processFrame = True
    
    # Create a while loop to consecutively read the image object from webcam
    while True:
        
        ret, frame = camera.read()
        
        if ret & frame is not None:
                    
            # Shrink the frame size of the captured image to 1/4 of original for the purpose of quicker face recognition
            smallFrame = cv2.resize(frame, (0, 0), fx = 0.25, fy = 0.25)
            
            # OpenCV uses BGR channels
            # face_recognition uses RGB channels
            # So the conversion should be peformed
            smallFrame = smallFrame[:, :, ::-1]
                
            # Only the handle the each frame in video for conserving time
            if processFrame:
                
                # Find the facial position and facial encoding from current video frame
                # locations = face_recognition.face_locations(smallFrame, model = "cnn")
                locations = face_recognition.face_locations(smallFrame)
                # Find the facial encodings
                encodings = face_recognition.face_encodings(smallFrame, locations)
                names = []
                # Iterate the facial image encoding because there could be multiple faces in the image

                for encoding in encodings:
                
                    # Compare the recognized face encoding to the known ones, to check whether they are the same person.
                    # There could be small discrepancy betwwen known face encoding and recognized face encoding. But whatever, it is good if the discrepancy is within a Fault Tolerance, default is 0.6
                    matches = face_recognition.compare_faces(knownFacialEncodings, encoding, tolerance = 0.35)
                    
                    name = "未知人员"
                    
                    # Grab the name if the face is recognized; Otherwise, just display the Unknown
                    if True in matches:
                        
                        firstMatch = matches.index(True)
                        name = knownFaceNames[firstMatch]
                        
                    names.append(name)
                    
            processFrame = not processFrame
            
            r, g, b, a = 0, 0, 255, 0
            
            # Iterate the names and discovered facial features, and draw on the image
            for (top, right, bottom, left), name in zip(locations, names):
                
                # When the face is being detected and recognizing, the image is resized into 1/4 of original one, now the resized image will be restored
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Draw a rectangle surrounding the face
                cv2.rectangle(frame, (left, top), (right, bottom), (r, g, b), 2)
                # Draw a solid heart rectangle to dispay the name of the person
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (r, g, b), cv2.FILLED)
                
                # font = cv2.FONT_HERSHEY_DUPLEX
                # cv2.putText(frame, name, (left + 6, bottom - 6), 1.0, (b, b, b), 1)
                
                # Add Chinese
                frame = DrawChinese(frame, name, (left + 15, bottom - 38), (b, b, b), 28)
                
            # Place the image on the video window
            cv2.imshow("Face Recognition", frame)
            # Allow exit by issuing the key "q" from keyboard
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
    # Release the object of webcam
    camera.release()
    # Dispose all the windows created by cv2
    cv2.destroyAllWindows()
        
# DrawFacialFeatures("../Inventory/Verification/jinhui_test.jpeg")
LiveRecognize(["../Inventory/Verification/Personnel_1_test.jpeg", "../Inventory/Verification/Personnel_2_test.jpg"])
