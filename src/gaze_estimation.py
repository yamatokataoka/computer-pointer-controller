import os
import sys
import logging as log
import numpy as np
import math
from openvino.inference_engine import IENetwork, IECore

class Gaze_Estimation:
    '''
    Class for the Gaze Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.6):
        '''
        set instance variables.
        '''
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.plugin = None
        self.network = None
        self.exec_network = None
        self.input_shape = None
        self.output_name = None
        self.threshold=threshold

    def load_model(self):
        ### Load the Inference Engine API
        self.plugin = IECore()

        ### Load the model
        model_xml = self.model_name
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        self.network = IENetwork(model=model_xml, weights=model_bin)

        ### Check model extesions and any unsupported layers
        self.check_model()

        ### Load the model network into a self.plugin variable
        self.exec_network = self.plugin.load_network(self.network, self.device)

        log.info("IR successfully loaded into Inference Engine.")

        ### Get the input and output information
        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_name = next(iter(self.network.outputs))

    def predict(self, eyes, head_pose_angles):
        '''
        This method is meant for running predictions on the input image.
        '''
        input_dict={'left_eye_image'  : eyes[0],
                    'right_eye_image' : eyes[1],
                    'head_pose_angles': head_pose_angles}
        self.exec_network.infer(input_dict)

        outputs = self.exec_network.requests[0].outputs[self.output_name]
        outputs = self.preprocess_output(outputs)

        return outputs

    def check_model(self):
        '''
        Check adding extensions and any unsupported layers
        '''
        ### Add a CPU extension, if applicable
        if self.extensions and "CPU" in self.device:
            self.plugin.add_extension(self.extensions, self.device)

        ### Check for supported layers
        supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)

        ### Check for any unsupported layers, and let the user
        ### know if anything is missing. Exit the program, if so.
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            log.warn("Unsupported layers found: {}".format(unsupported_layers))
            log.warn("Check whether extensions are available to add to IECore.")
            exit(1)

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        preprocess the output.
        '''
        # result is in orthogonal coordinate system (x,y,z. not yaw,pitch,roll)and not normalized
        outputs = outputs[0]

        # normalize the gaze vector
        gaze_vec_norm = outputs / np.linalg.norm(outputs)

        return gaze_vec_norm
