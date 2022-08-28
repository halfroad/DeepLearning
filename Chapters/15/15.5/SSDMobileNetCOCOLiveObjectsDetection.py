# Import the correcponding libraries
import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
import cv2

from six.moves.urllib import request
sys.path.append("../Exclusion/models/research/")

# Import the utils on Objects Detection
from object_detection.utils import ops as utilsOps, label_map_util as labelMapUtil, visualization_utils as visualizationUtils

# Revert back the matplotlib to TkAgg because in Utils, the Agg is used for console output
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
    
    print("Download file from {}".format(url))
    
    path = folder + fileName
    
    response = request.urlretrieve(url, path, ProgressReportCallback)
    
    print("response = {}".format(response))
    
    # Uncompress the downloaded model
    tar = tarfile.open(path)
    
    for member in tar.getmembers():
        
        entry = os.path.basename(member.name)
        
        if "frozen_inference_graph.pb" in entry:
            
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
    
    labelMap = labelMapUtil.load_labelmap(labels)
    categories = labelMapUtil.convert_label_map_to_categories(labelMap, max_num_classes = classificationsNumber, use_display_name = True)
    indexes = labelMapUtil.create_category_index(categories)
    
    i = 0
    
    for k, v in indexes.items():
        
        print("Index = {} - Category: {}".format(k, v))
        
        i += 1
        
    return indexes
    
def Capture(graph, categories):
    
    # Initialize the camera
    videoCapture = cv2.VideoCapture(0)
    
    if not videoCapture.isOpened:
        
        print("Cannot open the camera")
        exit()
    
    # Acquire the default graph of TensorFlow
    with graph.as_default():
        
        # Acquire the session of TensorFlow
        with tf.compat.v1.Session(graph = graph) as sess:
            
            # Iterate the frames from camera infinitely
            ret = True
            
            while(ret):
                
                # Read the frame from camera
                ret, frame = videoCapture.read()
                
                if ret and frame is not None:
                    
                    # Since the expected dimensions of image is 4, shape = [1, None, None, 3]. So here the image is expanded to 4 dimensions
                    expandedFrame = np.expand_dims(frame, axis = 0)
                    # Get the tensor of image
                    imageTensor = graph.get_tensor_by_name("image_tensor:0")
                    
                    # Get the tensor of detected bounaries from frame
                    boxes = graph.get_tensor_by_name("detection_boxes:0")
                    # Get the tensor of detected confidential scores from frame. The scores would be appeared alongwith category label.
                    scores = graph.get_tensor_by_name("detection_scores:0")
                    # Get the tensor of detected objects from frame
                    classes = graph.get_tensor_by_name("detection_classes:0")
                    # Get the tensor of detected number of objects
                    num_detections = graph.get_tensor_by_name("num_detections:0")
                    
                    # Begin the objects detection
                    (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict = {imageTensor: expandedFrame})
                    
                    # Visualize the detection result on the frame
                    
                    visualizationUtils.visualize_boxes_and_labels_on_image_array(frame,
                                                                                 np.squeeze(boxes),
                                                                                 np.squeeze(classes).astype(np.int32),
                                                                                 np.squeeze(scores),
                                                                                 categories,
                                                                                 use_normalized_coordinates = True,
                                                                                 line_thickness = 1)
                    
                    # Show the result on window
                    cv2.imshow("Live Obejcts Detection from Camera", cv2.resize(frame, (960, 7000)))
                    
                    #  Exit the application when keycode "q" is pressed
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        
                        cv2.destroyAllWindows()
                        videoCapture.release()
                        
                        break
                    
                    
    
def Start():
    
    name = "ssd_mobilenet_v1_coco_2018_01_28"
    
    exclusion = "../Exclusion/"
    folder = exclusion + "Downloads/" + name
    graphPath = folder + "/frozen_inference_graph.pb"
    
    if not os.path.exists(graphPath):
       #  shutil.rmtree(folder)
       graphPath = DownloadModel(name)
    
    graph = ExtractGraph(graphPath)
    categories = AcquireClassifications("mscoco_complete_label_map.pbtxt")

    Capture(graph, categories)

print("tensorflow version: {}".format(tf.__version__))

Start()