# STRAPS-3DHumanShapePose

## Installation

### Requirements
- Linux or macOS
- Python ≥ 3.6

### Instructions
We recommend using a virtual environment to install relevant dependencies. After creating a virtual environment, first install torch and torchvision: `pip install torch==1.4.0 torchvision==0.5.0`

Then install (my fork of) detectron2 and its dependencies (cython and pycocotools):
```
pip install cython
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install 'git+https://github.com/akashsengupta1997/detectron2.git'
```

The remaining dependencies can be installed by simply running: `pip install -r requirements.txt`. This will be sufficient for inference. If you wish run model training, you will require the PyTorch port of Neural Mesh Renderer: `pip install neural_renderer_pytorch==1.1.3`.

### Additional files
You will need to download the SMPL model. The [neutral model](http://smplify.is.tue.mpg.de) is required for training and running the demo code. If you want evaluate the model on datasets with gendered SMPL labels (such as 3DPW and SSP-3D), the male and female models are available [here](http://smpl.is.tue.mpg.de). You will need to convert the SMPL model files to be compatible with python3 by removing any chumpy objects. To do so, please follow the instructions [here](https://github.com/vchoutas/smplx/tree/master/tools).

Download required additional files here:  Place both the SMPL model and the additional files in the `additional` directory such that they have the following structure:

    STRAPS-3DHumanShapePose
    ├── additional                                      # Folder with additional files
    │   ├── smpl
    │       ├── SMPL_NEUTRAL.pkl                  # Gender-neutral SMPL model 
    │   ├── cocoplus_regressor.npy                # Cocoplus joints regressor
    │   ├── J_regressor_h36m.npy                  # Human3.6M joints regressor
    │   ├── J_regressor_extra.npy                 # Extra joints regressor
    │   ├── neutral_smpl_mean_params_6dpose.npz   # Mean gender-neutral SMPL parameters
    │   ├── smpl_faces.npy                        # SMPL mesh faces
    │   ├── cube_parts.npy
    │   └── vertex_texture.npy                    # etc.
    └── ...

### Model checkpoints.
Download pre-trained model checkpoints for our SMPL regressor, as well as for PointRend and DensePose (via detectron2) from here:  Place these files in a checkpoints directory, like so:

    STRAPS-3DHumanShapePose
        ├── checkpoints                               # Folder with model checkpoints
        │   ├── densepose_rcnn_R_101_fpn_s1x.pkl
        │   ├── pointrend_rcnn_R_50_fpn.pkl
        │   └── straps_model_checkpoint.tar           
        └── ...
  
