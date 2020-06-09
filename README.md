## cards_detection.ipynb is the final notebook with all the instructions of merging all the files

Whole dataset and files are stored on google drive and operations carried out by mounting the drive.

Since we are using Tensorflow Object Detection API, therefore download the tensorflow repository (https://github.com/tensorflow/models/tree/079d67d9a0b3407e8d074a200780f3835413ef99) into drive.

Also download the already trained faster_rcnn_inception_v2_coco_2018_01_28 model from http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz ,
we are gonna use the checkpoints (storing the trained parameter values).

# Directory structure

We have to upload the files (available in this repository) onto drive containing the tensorflow repository.

Upload:
### faster_rcnn_inception_v2_coco_2018_01_28
### xml_to_csv.py
### generate_tfrecord.py
### Object_detection_image.py
and create a folder "training",
in the directory structure of tensorflow repository - models/research/object_detection/

Upload:
### faster_rcnn_inception_v2_pets.config
### lebelmap.pbtxt
in the directory structure of tensorflow repository - models/research/object_detection/training/

