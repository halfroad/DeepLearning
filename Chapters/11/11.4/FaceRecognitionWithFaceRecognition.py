# Detect the features of face
from PIL import Image, ImageDraw
import face_recognition
import cv2

def DrawFacialFeatures(path):
    
    # Load the image into an array of NumPy
    image = face_recognition.load_image_file(path)
    # Query all the faces from image
    faceLandmarks = face_recognition.face_landmarks(image)
    
    print("{} faces observed".format(faceLandmarks))
    
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
    facialEncoding = face_recognition.face_encodings(image)[0]
    
    return facialEncoding

def LiveRecognize(path):
    
    # Grab the object of realtime webcab object
    camera = cv2.VideoCapture()
    camera.open(0, cv2.CAP_DSHOW)
    
    if not camera.isOpened():
        
        print("Cannot open camera")
        exit()
    
    # Load the known facial object
    facialEncoding = LoadFacialImage(path)
    
    # Create known facial encoding array
    knownFacialEncodings = [facialEncoding]
    # Create the name of the known facial object
    knownFaceNames = ["Zhang Qiang"]
    
    processFrame = True
    
    # Create a while loop to consecutively read tge image object from webcam
    while True:
        
        ret, frame = camera.read()
        
        print("ret = {}, frame = {}".format(ret, frame))
        
        # Shrink the frame size of the captured image to 1/4 of original for the purpose of quicker face recognition
        smallFrame = cv2.resize(frame, (0, 0), fx = 0.25, fy = 0.25)
        
        # OpenCV uses BGR channels
        # face_recognition uses RGB channels
        # So the conversion should be peformed
        smallFrame = smallFrame[:, :, :: -1]
        
        # Only the handle the each frame in video for conserving time
        if processFrame:
            
            # Find the facial position and facial encoding from current video frame
            locations = face_recognition.face_locations(smallFrame)
            encodings = face_recognition.face_encodings(smallFrame, locations)
            names = []
            
            # Iterate the facial image encoding because there could be multiple faces in the image
            for encoding in encodings:
                
                # Compare the recognized face encoding to the known ones, to check whether they are the same person.
                # There could be small discrepancy betwwen known face encoding and recognized face encoding. But whatever, it is good if the discrepancy is within a Fault Tolerance, default is 0.6
                matches = face_recognition.compare_faces(knownFacialEncodings, encoding)
                
                name = "Unknown"
                
                # Grab the name if the face is recognized; Otherwise, just display the Unknown
                if True in matches:
                    
                    firstMatch = matches.index(True)
                    name = knownFaceNames[firstMatch]
                    
                names.append(name)
                
        processFrame = not processFrame
        
        # Iterate the names and discovered facial features, and draw on the image
        for (top, right, bottom, left), name in zip(locations, names):
            
            # When the face is being detected and recognizing, the image is resized into 1/4 of original one, now the resized image will be restored
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Draw a rectangle surrounding the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a solid heart rectangle to dispay the name of the person
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            
            font = cv2.FONT_HERSHEY_DUPEX
            
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Place the image on the video window
        cv2.imshow("Video", frame)
        # Allow exit by issuing the key "q" from keyboard
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    # Release the object of webcam
    videoCapture.release()
    # Dispose all the windows created by cv2
    cv2.destroyAllWindows()
        

# DrawFacialFeatures("../Inventory/Verification/victor_test.jpeg")
LiveRecognize("../Inventory/Verification/victor_test.jpeg")