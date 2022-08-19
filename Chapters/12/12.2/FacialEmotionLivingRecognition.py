from PIL import ImageFont, ImageDraw, Image
from moviepy.editor import VideoFileClip

import cv2
import numpy as np
import sys

sys.path.append("../12.1/")

from FacialEmotionRecognition import LoadModel, classifications, width, height
model = LoadModel()

# Load the configuration file of cv2
haarcascade = "haarcascade_frontalface_default.xml"
cascadeClassifier = cv2.CascadeClassifier(haarcascade)

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
    path = "../../../Universal/Fonts/Songti.ttc"
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

def EditClip(originalVideo, destinationVideo, beginTime = 0, endTime = 10):
    
    # Specify the path of video, and specify the begin time and end time for the edit clip via method suclip(t_start, t_end)
    videoFileClip = VideoFileClip(originalVideo).subclip(beginTime, endTime)
        
    # Edit each frame via the method ProcessFrame, and update the video after the frame is edited
    whiteClip = videoFileClip.fl_image(ProcessFrame)
    
    # Export the edited video into the destination path, and show the percertage of completion
    whiteClip.write_videofile(destinationVideo)

counter = 0

originalVideo = "../Inventory/Videos/video.mp4"
destinationVideo = "../Inventory/Videos/video_edited.mp4"

EditClip(originalVideo, destinationVideo)
    