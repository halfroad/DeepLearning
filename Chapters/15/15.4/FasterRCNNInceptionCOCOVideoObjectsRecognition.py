import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
import glob
from moviepy.editor import VideoFileClip

from urllib import request
from PIL import Image

# Import the superior folder for performing the modules
sys.path.append("../Exclusion/models/research/")

# Import the uitls module under Object Detection
from object_detection.utils import ops as utilsOps, label_map_util as labelMapUtil, visualization_utils as visualizationUtils

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def DownloadModel(name):
    
    # Name of model and download URLs
    fileName = name + ".tar.gz"
    folder = "../Exclusion/Downloads/"
    
    # Frozen folder Graph after the model gets downloaded and uncompressed. The architecture of pretrained model is stored in Graph
    graphPath = folder + name + "/frozen_inference_graph.pb"
    
    downloadBaseURL = "http://download.tensorflow.org/models/object_detection/"
    
    # Download the model
    url = downloadBaseURL + fileName
    
    print("Dowload file from {}".format(url))
    
    path = folder + fileName
    
    response = request.urlretrieve(url, path, ProgressReportCallback)
    
    print("response = {}".format(response))
    
    # Uncompress the downloaded model
    tar = tarfile.open(path)
    
    for member in tar.getmembers():
        
        fileName = os.path.basename(member.name)
        
        if "frozen_inference_graph.pb" in fileName:
            
            # tar.extract(file, os.getcwd())
            tar.extract(member, folder)
            
    return graphPath

def ProgressReportCallback(count, blockSize, totalSize):
    
    print("Count = {}, Block Size = {}, Total Size = {}, Percentage = {:.2f}%".format(count, blockSize, totalSize, 100 * (count * blockSize) / totalSize))
    
def ExtractGraph(graphPath):
    
    graph = tf.Graph()
    
    with graph.as_default():
        
        odGraphDef = tf.compat.v1.GraphDef()
        
        with tf.io.gfile.GFile(graphPath, "rb") as gf:
            
            serializedGraph = gf.read()
            
            odGraphDef.ParseFromString(serializedGraph)
            
            tf.import_graph_def(odGraphDef, name = "")
            
            return graph
    
def AcquireClassifications(name):
    
    # mscoco_label_pbtxt stores the mapping relationship between classifications and indexes
    labels = os.path.join("../Exclusion/models/research/object_detection/data", name)
    classificationsNumber = sys.maxsize
    
    labelsMap = labelMapUtil.load_labelmap(labels)
    categories = labelMapUtil.convert_label_map_to_categories(labelsMap, max_num_classes = classificationsNumber, use_display_name = True)
    indexes = labelMapUtil.create_category_index(categories)
    
    i = 0
    
    for k, v in indexes.items():
        
        print("Index: {} - Category: {}".format(k, v))
        
        i += 1
        
    return indexes

def Infer(image, sess, graph, categories):
    
    # Acquire the graph
    with graph.as_default():
        
        # Acquire the handlers of tensors of input and output
        ops = tf.compat.v1.get_default_graph().get_operations()
        
        # Get all the names of tensors
        tensorNames = {output.name for op in ops for output in op.outputs}
        tensorDictionary = {}
        
        for key in ["num_detections", "detection_boxes", "detection_scores", "detection_classes", "detection_masks"]:
            
            tensorName = key + ":0"
            
            if tensorName in tensorNames:
                
                tensorDictionary[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(tensorName)
                
        if "detection_masks" in tensorDictionary:
            
            # Handle the single image
            detectionBoxes = tf.squeeze(tensorDictionary["detection_boxes"], [0])
            detectionMasks = tf.squeeze(tensorDictionary["detection_masks"], [0])
            
            detectionsNumber = tf.cast(tensorDictionary["num_detections"][0], tf.int32)
            detectionBoxes = tf.slice(detectionBoxes, [0, 0], [detectionsNumber, -1])
            detectionMasks = tf.slice(detectionMasks, [0, 0, 0], [detectionsNumber, -1, -1])
            
            # Reframe to convert the mask from frame coordinate to image coordinate, and fit the size of image
            reframedDetectionMasks = utilsOps.reframe_box_masks_to_image_masks(detectionMasks, detectionBoxes, image.shape[0], image.shape[1])
            reframedDetectionMasks = tf.cast(tf.greater(reframedDetectionMasks, 0.5), tf.uint8)
            
            # Add the batch dimensions to conform the norms
            tensorDictionary["detection_masks"] = tf.expand_dims(reframedDetectionMasks, 0)
            
        imageTensor = tf.compat.v1.get_default_graph().get_tensor_by_name("image_tensor:0")
        
        # Perform the object inference, real detection
        outputDictionary = sess.run(tensorDictionary, feed_dict = {imageTensor: np.expand_dims(image, 0)})

        # All the outputs are float32 NumPy array, conversion is needed
        # num_detections means the number of detection boxs
        outputDictionary["num_detections"] = int(outputDictionary["num_detections"][0])
        
        # detection_classes means the class of detected box
        outputDictionary["detection_classes"] = outputDictionary["detection_classes"][0].astype(np.uint8)
        
        # detection_boxes means the detected box
        outputDictionary["detection_boxes"] = outputDictionary["detection_boxes"][0]
        
        # detection_scores means the detected detection scores
        outputDictionary["detection_scores"] = outputDictionary["detection_scores"][0]
        
        if "detection_masks" in outputDictionary:
            
            outputDictionary["detection_masks"] = outputDictionary["detection_masks"][0]
                        
        # Reflect the detected boundaries of object on the image
        # imageCopy = np.copy(image)
        visualizationUtils.visualize_boxes_and_labels_on_image_array(image,
                                                                     outputDictionary["detection_boxes"],
                                                                     outputDictionary["detection_classes"],
                                                                     outputDictionary["detection_scores"],
                                                                     categories,
                                                                     instance_masks = outputDictionary.get("detection_masks"),
                                                                     use_normalized_coordinates = True,
                                                                     line_thickness = 1)
        
        return image
    
def DetectObjects(graph, categories, image):
    
    global counter

    if counter % 1 == 0:
        
        with graph.as_default():
            
            with tf.compat.v1.Session(graph = graph) as sess:
                
                image = Infer(np.array(image), sess, graph, categories)
                
    counter += 1
    
    return image

def EditVideo(graph, categories, path, clipped = False, beginTime = 0, endTime = 10):
        
    # Video path of output
    fileName, extension = os.path.splitext(path)
    editedVideo = fileName + "_objects_detected" + extension
    
    counter = 0
    
    if clipped:
        
        # Specify the address of the vidoe, and specify the ebgin and end time for the method subclip()
        videoFileClip = VideoFileClip(path).subclip(beginTime, endTime)
        
        # Callback handler DetectObjects to edit the video for each frame, and reflect the updated image on the video
        clip = videoFileClip.fl_image(lambda image: DetectObjects(graph, categories, image))
        
        # Export the video into local disk, and show the percentage of cimpletion
        clip.write_videofile(editedVideo)
        
    else:
        
        # Specify the address of the vidoe, and specify the ebgin and end time for the method subclip()
        videoFileClip = VideoFileClip(path)
        
        # Callback handler DetectObjects to edit the video for each frame, and reflect the updated image on the video
        clip = videoFileClip.fl_image(lambda image: DetectObjects(graph, categories, image))
        
        # Export the video into local disk, and show the percentage of cimpletion
        clip.write_videofile(editedVideo)
        
def Start():
    
    name = "faster_rcnn_inception_v2_coco_2018_01_28"
    
    exclusion = "../Exclusion/"
    folder = exclusion + "Downloads/" + name
    graphPath = folder + "/frozen_inference_graph.pb"
    
    if not os.path.exists(graphPath):
       #  shutil.rmtree(folder)
       graphPath = DownloadModel(name)
    
    graph = ExtractGraph(graphPath)
    categories = AcquireClassifications("mscoco_complete_label_map.pbtxt")

    EditVideo(graph, categories, "../Exclusion/Videos/The_Bund.mp4", True, 0, 1)

print("tensorflow version: {}".format(tf.__version__))

counter = 0

Start()
