from PIL import ImageFont, ImageDraw, Image
from moviepy.editor import VideoFileClip

import cv2
import numpy as np
import sys

sys.path.append("../12.1/")

from FacialEmotionRecognition import LoadModel, width, height
model = LoadModel()

# Load the configuration file of cv2
haarcascade = "haarcascade_frontalface_default.xml"
cascadeClassifier = cv2.CascadeClassifier(haarcascade)
classifications = ["愤怒", "厌恶", "害怕", "高兴", "悲伤", "惊喜", "一般"]

def PredictEmotion(frame, horizonzal, vertical, width, height, classifications, model):
    
    # Recognize the emotion from face using model
    # Only read the face area recongized by model
    crop = frame[vertical: vertical + height, horizonzal: horizonzal + width]
    
    # Resize the face into 48 * 48
    crop = cv2.resize(crop, (48, 48))
    # Convert the crop into gray scale
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Convert the image into values between 0 ~ 1
    crop = crop.astype(np.float32) / 255
    crop = np.asarray(crop)
    
    # Reshape the crop with (batch_size, height, width, channels)
    crop = crop.reshape(1, crop.shape[0], crop.shape[1], 1)
        
    possibilities = model.predict(crop)
    
    # Recognize the classification of emotion
    emotion = classifications[np.argmax(possibilities)]
    
    return emotion

def RedrawImage(frame,  emotion, horizontal, vertical):
    
    # Load the font
    # path = "../../../Universal/Fonts/Songti.ttc"
    path = "Songti.ttc"
    font = ImageFont.truetype(path, 32)
    
    # Sythesize the orginal frame and emtion to new image
    # Convert the original frame into RGB
    pilImage = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Create the object of drawable image
    drawable = ImageDraw.Draw(pilImage)
    
    # Draw the text on the corrdinate (x, y)
    drawable.text((horizontal, vertical - 50), emotion, font = font, fill = (32, 183, 228))
    
    # After the frame and emotion sythesized, then convert the image into BGR
    image = cv2.cvtColor(np.asarray(pilImage), cv2.COLOR_RGB2BGR)
    
    # Return the sythesized and converted image
    return image

def ProcessFrame(frame):
    
    global counter
    
    if counter % 1 == 0:
    
        finalImage = frame
        
        # Gray out the image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Recognize the faces via cv2 (More than 1 faces can be detected)
        faces = cascadeClassifier.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30))
        
        # Iterate the faces
        for (x, y, w, h) in faces:
            
            # Draw the recognized face on the original frame with rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (228, 32, 183), 2)
            
            # Result of face recognition
            emotion = PredictEmotion(frame, x, y, w, h, classifications, model)
            
            # Sythesize the frame
            finalImage = RedrawImage(frame, emotion, x, y)
            
    counter += 1
    
    # Return the new csythesized image
    return finalImage
        
def EditClip(originalVideo, destinationVideo, clip = False, beginTime = 0, endTime = 10):
    
    if clip:
    
        # Specify the path of video, and specify the begin time and end time for the edit clip via method suclip(t_start, t_end)
        videoFileClip = VideoFileClip(originalVideo).subclip(beginTime, endTime)
            
        # Edit each frame via the method ProcessFrame, and update the video after the frame is edited
        whiteClip = videoFileClip.fl_image(ProcessFrame)
        
        # Export the edited video into the destination path, and show the percertage of completion
        whiteClip.write_videofile(destinationVideo, audio_codec = "aac")
        
    else:
    
        # Specify the path of video, and specify the begin time and end time for the edit clip via method suclip(t_start, t_end)
        videoFileClip = VideoFileClip(originalVideo)
            
        # Edit each frame via the method ProcessFrame, and update the video after the frame is edited
        whiteClip = videoFileClip.fl_image(ProcessFrame)
        
        # Export the edited video into the destination path, and show the percertage of completion
        # whiteClip.write_videofile(destinationVideo, audio = True)
        
        whiteClip.write_videofile(destinationVideo, audio_codec = "aac")
        
def LiveRecognize():
    
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        
        print("Cannot open the camera")
        exit()
        
    processFrame = True
    
    # Create a while loop to consecutively read the image object from webcam
    while True:
        
        # Obtain the frames one by one fro video streaming
        ret, frame = camera.read()
        
        # Convert the frame (image) into gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Recognize the faces from frame
        faces = cascadeClassifier.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30))
        
        # Iterate the array of faces
        for (x, y, w, h) in faces:
            
            # Draw the facial rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (288, 32, 183), 2)
            # Recognize the emotion of the face
            emotion = PredictEmotion(frame, x, y, w, h, classifications, model)
            # Sythesize the frame and emotion string on a new image
            _image = RedrawImage(frame, emotion, x, y)
            # Show the sythesized image on the window, and resize the image into 800 * 500
            cv2.imshow("Facial Emotion Recognition", cv2.resize(800, 500))
            
        # Exit the application when key "q" pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    camera.release()
    cv2.destroyAllWindows()

counter = 0

'''

originalVideo = "../Inventory/Videos/video3.mp4"
destinationVideo = "../Inventory/Videos/video3_edited3.mp4"

EditClip(originalVideo, destinationVideo)

'''

LiveRecognize()
