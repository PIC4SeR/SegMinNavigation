# Row Crop Follow

- [Row Crop Follow](#row-crop-follow)
  - [Nodes](#nodes)
    - [seg\_controller\_node](#seg_controller_node)
      - [Topics](#topics)
      - [Parameters](#parameters)
    - [otsu\_thresholding\_node](#otsu_thresholding_node)
      - [Topics](#topics-1)
      - [Parameters](#parameters-1)
  - [Usage](#usage)
    - [Requirements:](#requirements)
    - [Installation](#installation)
    - [Run the application](#run-the-application)


## Nodes
### seg_controller_node
This node can run the controllers SegZeros, SegMin and SegMinD described in the paper **GPS-free Autonomous Navigation in Cluttered Tree Rows with Deep Semantic Segmentation**.

#### Topics
Set topic names through their parameter
- Subscriptions:
  - rgb_topic
  - depth_topic
- Publishers:
  - cmd_vel

#### Parameters
`controller_type`: SegZeros, SegMin or SegMinD

`cmd_vel_topic`, `rgb_topic`, `depth_topic`

`model_file`: full path to the model file

`inference_device`: could be either cpu or gpu. For TFLite models cpu is required.

`normalization`: normalization used for DNN  input preprocessing. Default: imagenet

`input_width_network`, `input_height_network`

`lin_vel_gain`, `ang_vel_gain`

`max_lin_vel`, `max_ang_vel`

`ema_filter_alpha`: exponential factor used in the Exponential Moving Average filter applied to smooth the velocities in output. Set to 1.0 to disable.

`depth_threshold`: maximum depth value in meters. Values over this threshold are saturated to this value.

`confidence_threshold`: confidence threshold for the prediction of the DNN

`show_result`, `publish_processed_image`

### otsu_thresholding_node

This is an implementation of the work "**Autonomous Navigation and Crop Row Detection in Vineyards Using Machine Vision with 2D Camera**"[^1] used for comparison.

> [^1]: Mendez, E.; Piña Camacho, J.; Escobedo Cabello, J.A.; Gómez-Espinosa, A. Autonomous Navigation and Crop Row Detection in Vineyards Using Machine Vision with 2D Camera. Automation 2023, 4, 309-326. [https://doi.org/10.3390/automation4040018](https://doi.org/10.3390/automation4040018)

#### Topics
Set topic names through their parameter
- Subscriptions:
  - rgb_topic
- Publishers:
  - cmd_vel
  - `~/result_img`

#### Parameters
`cmd_vel_topic`, `rgb_topic`

`propotional_criterion`, `deviation_control_gain`, `correction_control_gain`: respectively PC, G and P as described in the original paper

`linear_velocity`

`grayscale_method`: Choices: cv2, 2gbr, gb. The two althernative methods described in the paper, 2GBR and GB, where both implemented and can be used. Additionally, it is provided the possibility of using the classical conversion provided by OpenCV.

`show_result`: whether to show the result of the control algorithm using OpenCV.

## Usage

### Requirements:
- ROS2 Humble
- TensorFlow 2.x (2.10 tested)

### Installation
- Clone the repo in your workspace with `git clone https://github.com/PIC4SeR/SegMinNavigation.git`
- Install the required dependencies with `rosdep install --from-path src --ignore-src -y`
- Build the workspace with `colcon build --symlink-install --packages-select row_crop_follow`

### Run the application

Two launch files are provided for convenience and can be found in the [launch](./launch/) folder. For both launches a parameter file can be set as launch argument using `params_file:=path-to-file`.

Example: 
```bash
 ros2 launch row_crop_follow seg_controller.launch.py params_file:=~/my_params.yaml 
```
