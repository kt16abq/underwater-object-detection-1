# Kesci Underwater Object Detection Algorithm Competition underwater object detection algorithm contest Baseline <font color=red>**A list mAP 48.7**</font><br />

## Competition address: [Kesci underwater target detection] (https://www.kesci.com/home/competition/5e535a612537a0002ca864ac)

## Please refer to the updated [repo](https://github.com/zhengye1995/kesci-2021-underwater-optics) for URPC 2021!!!

## Update Tried what doesn't work:

+ data augmentation
   + flip rotation
   + color cast calibration
   + Brightness, contrast enhancement
   + Various blur smoothing operators
   + Defog
   + mixup
   + Introduce data from previous years
+ model integration
   + direct nms
   + weighted nms
   + wbf
+ training samples
   + Based on data distribution, i.e. domain sampling
   + OHEM
+ model section
   + DCN
   + se154 is close to x101
+ Error analysis: From the difficulty of rpn convergence and the serious loss of OHEM points, it can be analyzed that the main sources of current prediction errors are:
   + High-scoring fp: A considerable part of it is caused by missing labels and wrong labels. For example, there are adjacent frames in the training set that have the same target marked in the previous frame and not marked in the next frame. Unite
   + Low-scoring tp: This category is mainly focused on fuzzy targets. The overall score of the model for fuzzy target prediction is low, but the data annotations are not consistent with the fuzzy target standards.
   + Missed detection: A few fuzzy small targets missed detection
## Update Update the online mAP of htc pre-trained resnext101 64x4d to **48.7**
## the whole idea
    + detection algorithm: Cascade R-CNN
    + backbone: ResNet50 + FPN
    + post process: soft nms
    + Based on [mmdetection](https://github.com/open-mmlab/mmdetection/), not the latest version, you can upgrade by yourself
    + Both res50 and se50 can reach online testA 46-47 mAP, and model integration can reach 49+ after [spytensor](https://github.com/spytensor) test
    + resnext101 64x4d 48.7mAP
## Code environment and dependencies

+ OS: Ubuntu16.10
+ GPU: 2080Ti * 4
+ python: python3.7
+ nvidia dependencies:
    - cuda: 10.0.130
    - cudnn: 7.5.1
    - nvidia driver version: 430.14
+ deeplearning framework: pytorch1.1.0
+ Please refer to requirement.txt for other dependencies
+ The number of graphics cards is not important, you can adjust the learning rate according to the multiple of the number of graphics cards you have

## Training data preparation

- **Corresponding folder creation preparation**

   - Create a new data folder under the root directory of the code, or create a soft link according to your own situation
   - Go to the data folder and create a folder:
  
      annotations

      pretrained

      results

      submit

   - Unzip the officially provided training and test data into the data directory to generate:
    
     train

     test-A-image
    
    
- **label file format conversion**

   - The official xml type label file in VOC format is provided. Personally, I am used to using COCO format for training, so format conversion is performed.
  
   - Use tools/data_process/xml2coco.py to convert the label file to COCO format, and the new label file train.json will be saved in the data/train/annotations directory

   - In order to facilitate the use of mmd multi-process testing (faster), we also generate a pseudo-label file for the test data, run tools/data_process/generate_test_json.py to generate testA.json, and the pseudo-label file will be saved in the data/train/annotations directory Down

   - Overall operation content:

     -python tools/data_process/xml2coco.py

     -python tools/data_process/generate_test_json.py

- **Pre-trained model download**
   - Download mmdetection's official open source casacde-rcnn-r50-fpn-2x COCO pre-training model [cascade_rcnn_r50_fpn_20e_20181123-db483a09.pth](https://open-mmlab.oss-cn-beijing.aliyuncs.com/mmdetection/models/ cascade_rcnn_r50_fpn_20e_20181123-db483a09.pth) and placed in the data/pretrained directory
   - For the pre-training of senet50, see: [mmd-senet](https://github.com/zhengye1995/pretrained), here I would like to thank [jsonc](https://github.com/jsnoc) for the pre-training training model
   - Download mmdetection's official open source htc [resnext 64x4d pre-training model] (https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20195068.9c)

## Dependency installation and compilation


- **depends on installation and compilation**

    1. Create and activate a virtual environment
         conda create -n underwater python=3.7 -y
         conda activate underwater

    2. Install pytorch
         conda install pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=10.0 -c pytorch
        
    3. Install other dependencies
         pip install cython && pip --no-cache-dir install -r requirements.txt
   
    4. Compile cuda op, etc.:
         python setup.py develop
   

## Model training and prediction
    
    - **train**

1. Run:
        
         r50:
        
chmod +x tools/dist_train.sh

         ./tools/dist_train.sh configs/underwater/cas_r50/cascade_rcnn_r50_fpn_1x.py 4
        
         se50:
        
chmod +x tools/dist_train.sh

         ./tools/dist_train.sh configs/underwater/cas_se/cas_se50_12ep.py 4
        
         x101_64x4d (htc pretrained):
        
chmod +x tools/dist_train.sh

         ./tools/dist_train.sh configs/underwater/cas_x101/cascade_rcnn_x101_64x4d_fpn_1x.py 4
        
         (The above 4 is the number of my gpu, please modify it yourself)

    2. Both the training process file and the final weight file are saved in the workdir directory specified in the config file

    - **predict**

     1. Run:
    
         r50:

         chmod +x tools/dist_test.sh

         ./tools/dist_test.sh configs/underwater/cas_r50/cascade_rcnn_r50_fpn_1x.py workdirs/cascade_rcnn_r50_fpn_1x/latest.pth 4 --json_out results/cas_r50.json

         (The above 4 is the number of my gpu, please modify it yourself)
        
         se50:

         chmod +x tools/dist_test.sh

         ./tools/dist_test.sh configs/underwater/cas_se/cas_se50_12ep.py workdirs/cas_se50_12ep/latest.pth 4 --json_out results/cas_se50.json

         (The above 4 is the number of my gpu, please modify it yourself)
        
         x101_64x4d (htc pretrained):

         chmod +x tools/dist_test.sh

         ./tools/dist_test.sh configs/underwater/cas_x101/cascade_rcnn_x101_64x4d_fpn_1x.py workdirs/cas_x101_64x4d_fpn_htc_1x/latest.pth 4 --json_out results/cas_x101.json


     2. The prediction result file will be saved in the /results directory

     3. Convert the mmd prediction result to submit a csv format file:
       
        python tools/post_process/json2submit.py --test_json cas_r50.bbox.json --submit_file cas_r50.csv
       
        python tools/post_process/json2submit.py --test_json cas_se50.bbox.json --submit_file cas_se50.csv
       
        python tools/post_process/json2submit.py --test_json cas_x101.bbox.json --submit_file cas_x101.csv

        The submission files cas_r50.csv, cas_se50.csv and cas_x101.csv that finally conform to the official format are located in the submit directory
    


