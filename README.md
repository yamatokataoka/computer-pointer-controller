# Computer Pointer Controller

**Computer Pointer Controller is a python application to control the mouse pointer with your eyes using machine learning.** It's using the [Gaze Estimation model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html) to estimate the gaze of the user's eyes and change the mouse pointer position accordingly

## Project Set Up and Installation
### Prerequisites
This project requires OpenVINO™ toolkit to optimize and run machine learning models.

Follow [the documentation](https://docs.openvinotoolkit.org/latest/index.html) for OpenVINO™ toolkit here.

The pre-trained models for OpenVINO™ toolkit is on the project root directory.

Install all the required Python modules on the project root directory.
```
pip install -r requirements.txt
```

## Demo
Try running on demo.mp4 on the project src directory.
```
python main.py
```

After finishing the inference, you can check output video named `out.mp4` on the root directory.

## Documentation
### The command line options
All arguments are optional.

| Argument          | Description                                | Default         |
|-------------------|--------------------------------------------|-----------------|
| -p / --input_path | The location of the input file             | ../bin/demo.mp4 |
| -t / --input_type | Type of input, any of cam, video and image | video           |
| -d / --device     | The device name                            | CPU             |
| -b                | Draw bounding boxes                        | False           |
| -g                | Draw gaze lines                            | False           |

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.
