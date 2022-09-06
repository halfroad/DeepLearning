'''

Apple 2 Orange

python3 build_data.py --X_input_dir data/apple2orange/trainA --Y_input_dir data/apple2orange/trainB -X_output_file ../Datasets/Apple2Orange/TFRecords/Apple.tfrecords --Y_output_file ../Datasets/Apple2Orange/TFRecords/Orange.tfrecords


python3 train.py --X ../Datasets/Apple2Orange/TFRecords/Apple.tfrecords --Y ../Datasets/Apple2Orange/TFRecords/Orange.tfrecords

python3 export_graph.py --checkpoint_dir ../Checkpoints/20220906-0859/ --XtoY_model Apple2Orange.pb --YtoX_model Orange2Apple.pb --image 256

python3 inference.py --model ../Models/Pretrained/Apple2Orange.pb --input data/apple2orange/testA/n07740461_41.jpg --output ../Outputs/n07740461_41_Tweaked.jpg


Horse 2 Zebra

python3 build_data.py --X_input_dir data/horse2zebra/trainA --Y_input_dir data/horse2zebra/trainB -X_output_file ../Datasets/Horse2Zebra/TFRecords/Horse.tfrecords --Y_output_file ../Datasets/Horse2Zebra/TFRecords/Zebra.tfrecords

python3 train.py --X ../Datasets/Horse2Zebra/TFRecords/Horse.tfrecords --Y ../Datasets/Horse2Zebra/TFRecords/Zebra.tfrecords

python3 export_graph.py --checkpoint_dir ../Checkpoints/Horse2Zebra/20220906-1044/ --XtoY_model Horse2Zebra.pb --YtoX_model Zebra2Horse.pb --image 256

python3 inference.py --model ../Models/Pretrained/Horse2Zebra.pb --input ../Images/horse.png --output ../Images/horse_to_zebra.png --image_size 256


http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2

python3 faceswap.py ../Images/Man2Woman/Victor_2.jpg ../Images/Man2Woman/77_female_2.jpg


'''

import os
import matplotlib.pyplot as plt

from PIL import Image

def PlotImage(path):
    
    image = Image.open(path)
    
    figure = plt.figure()
    
    plt.imshow(image)
    plt.grid(False)
    plt.show()

PlotImage("../Exclusion/CycleGAN-TensorFlow-1/data/apple2orange/testA/n07740461_41.jpg")
PlotImage("../Exclusion/Outputs/n07740461_41_Tweaked.jpg")

PlotImage("../Exclusion/CycleGAN-TensorFlow-1/data/apple2orange/testB/n07749192_401.jpg")
PlotImage("../Exclusion/Outputs/n07749192_401_Tweaked.jpg")
