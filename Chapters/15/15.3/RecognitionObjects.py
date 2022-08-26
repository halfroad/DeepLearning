import numpy as np
import shutil
import os
import sys
import tarfile
import tensorflow as tf
import zipfile
import glob

from cv2 import imread
from urllib import request
from collections import defaultdict
from io import StringIO

print("tensorflow {}".format(tf.__version__))

# Import the superior folder for performing the modules
sys.path.append("../Exclusion/models/research/")

from object_detection.utils import ops as utilOps

# Import the uitls module under Object Detection
from object_detection.utils import label_map_util as labelMapUtil
from object_detection.utils import visualization_utils as visualizationUtils

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

DIRECTORY = "../Exclusion/Downloads/"
EXTENSION = ".tar.gz"

 # mscoco_label_pbtxt stores the classifications and mapping relation of index
labels = os.path.join("../Exclusion/models/research/object_detection/data", "mscoco_label_map.pbtxt")
    
classificationsNumber = 90

def DownloadModel(name):
    
    # Name of model and download URLs
    fileName = name + EXTENSION
    
    downloadBaseURL = "http://download.tensorflow.org/models/object_detection/"
    
    # Freeze the folder Graph after the model gets downloaded and uncompressed. The architecture of pretrained model is stored in Graph
    frozenGraph = DIRECTORY + name + "/" + "frozen_inference_graph.pb"
    
    # Download the model
    url = downloadBaseURL + fileName
    
    print("Dowload file from {}".format(url))
    
    path = DIRECTORY + fileName
    
    reponse = request.urlretrieve(url, path, ProgressReportHook)
    
    # Uncompress the downloaded model
    tar = tarfile.open(path)
    
    for file in tar.getmembers():
        
        name = os.path.basename(file.name)
        
        if "frozen_inference_graph.pb" in name:
            
            # tar.extract(file, os.getcwd())
            tar.extract(file, DIRECTORY)
            
    return DIRECTORY + frozenGraph, labels

def ProgressReportHook(count, blockSize, totalSize):
    
    print("count = {}, blockSize = {}, totalSize = {}".format(count, blockSize, totalSize))
            
def ExtractGraph(frozenGraph):
    
    detectionGraph = tf.Graph()
    
    with detectionGraph.as_default():
        
        odGraphDef = tf.compat.v1.GraphDef()
        
        with tf.io.gfile.GFile(frozenGraph, "rb") as gf:
            
            serializedGraph = gf.read()
            
            odGraphDef.ParseFromString(serializedGraph)
            tf.import_graph_def(odGraphDef, name = "")
            
            return detectionGraph
            

def AcquireClassifications():
    
    labelsMap = labelMapUtil.load_labelmap(labels)
    categories = labelMapUtil.convert_label_map_to_categories(labelsMap, max_num_classes = classificationsNumber, use_display_name = True)
    
    indexes = labelMapUtil.create_category_index(categories)
    
    i = 0
    
    for k, v in indexes.items():
        
        print("Index: {} - Category: {}".format(k, v))
        
        i += 1
        
        if i == 10:
            
            break
        
    return indexes

def Infer(image, graph):
    
    # Get the graph image
    with graph.as_default():
        
        # Iniatiate a TensorFlow session
        with tf.compat.v1.Session() as sess:
            
            # Get the handlers of tensors of input and output
            ops = tf.compat.v1.get_default_graph().get_operations()
            
            # Get all the names of tensors
            tensorNames = {output.name for op in ops for output in op.outputs}
                        
            tensorDictionary = {}
            
            for key in ["num_detections", "detection_boxes", "detection_scores", "detection_classes", "detection_masks"]:
                
                name = key + ":0"
                
                if name in tensorNames:
                    
                    tensorDictionary[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(name)
                    
            if "detection_masks" in tensorDictionary:
                
                # Handle the single image
                detectionBoxes = tf.squeeze(tensorDictionary["detection_boxes"], [0])
                detectionMasks = tf.squeeze(tensorDictionary["detection_masks"], [0])
                
                # Reframe to convert the mask from frame coordinate to image coordinate, and fit the size of image
                detectionNumber = tf.cast(tensorDictionary["num_detections"][0], tf.int32)
                
                detectionBoxes = tf.slice(detectionBoxes, [0, 0], [detectionNumber, -1])
                detectionMasks = tf.slice(detectionMasks, [0, 0, 0], [detectionNumber, -1, -1])
                
                reframedDetectionMasks = utilOps.reframe_box_masks_to_image_masks(detectionMasks, detectionBoxes, image.shape[0], image.shape[1])
                reframedDetectionMasks = tf.cast(tf.greater(reframedDetectionMasks, 0.5), tf.uint8)
                
                # Add the batch dimensions to conform the norms
                tensorDictionary["detection_masks"] = tf.expand_dims(reframedDetectionMasks, 0)
    
            imageTensor = tf.compat.v1.get_default_graph().get_tensor_by_name("image_tensor:0")
            
            # Perform the object inference, real detection
            outputDictionary = sess.run(tensorDictionary, feed_dict = {imageTensor: np.expand_dims(image, 0)})
            
            # All the outputs are float32 NumPy array, conversion is needed
            # detectionNumber means the number of detection boxs
            outputDictionary["num_detections"] = int(outputDictionary["num_detections"][0])
            
            # detectionClasses means the class of detected box
            outputDictionary["detection_classes"] = outputDictionary["detection_classes"][0].astype(np.uint8)
            
            # detectionBoxes means the detected box
            outputDictionary["detection_boxes"] = outputDictionary["detection_boxes"][0]
            
            # detectionScores means the detected detection scores
            outputDictionary["detection_scores"] = outputDictionary["detection_scores"][0]
            
            if "detection_masks" in outputDictionary:
                
                outputDictionary["detection_masks"] = outputDictionary["detection_masks"][0]
                
            return outputDictionary

def DetectImages():

    frozenGraph = DIRECTORY + name + "/" + "frozen_inference_graph.pb"
    
    detectionGraph = ExtractGraph(frozenGraph)
    categories = AcquireClassifications()
    
    paths = glob.glob("../Exclusion/Images/*.jpeg")
    
    imageSize = (12, 8)
    
    for path in paths:
                
        # Comvert the image into NumPy array
        array = imread(path, 1)
        
        # Since the model needs the shape [1, None, None, 3], the dimensions needs to expand
        expandedArray = np.expand_dims(array, axis = 0)
        
        # Perform the detection
        outputDictionary = Infer(array, detectionGraph)
        
        # Visualize the detection
        visualizationUtils.visualize_boxes_and_labels_on_image_array(array, outputDictionary["detection_boxes"], outputDictionary["detection_classes"], outputDictionary["detection_scores"], categories, instance_masks = outputDictionary.get("detection_masks"), use_normalized_coordinates = True, line_thickness = 2)
        
        plt.figure(figsize = imageSize)
        plt.imshow(array)
        
        plt.show()
        
name = "mask_rcnn_inception_v2_coco_2018_01_28"

'''

folder = DIRECTORY + name

if os.path.exists(folder):
    shutil.rmtree(folder)
    
fileName = DIRECTORY + name + EXTENSION

if os.path.exists(fileName):
    os.remove(fileName)

frozenGraph = DownloadModel(name)

AcquireClassifications()

'''

DetectImages()
