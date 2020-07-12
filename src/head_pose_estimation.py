import os
import sys
import logging as log
import cv2
from openvino.inference_engine import IENetwork, IECore

class Head_Pose_Estimation:
    '''
    Class for the Head Pose Estimation Model.
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
        self.input_name = None
        self.input_shape = None
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

        ### Get the input information
        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape

    def predict(self, image):
        '''
        This method is meant for running predictions on the input image.
        '''
        input_image = self.preprocess_input(image)

        input_dict={self.input_name:input_image}
        outputs = self.exec_network.infer(input_dict)

        yaw, pitch, roll = self.preprocess_output(outputs)

        return yaw, pitch, roll

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

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        preprocess image.
        '''
        p_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        return p_frame

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        preprocess the output.
        '''
        yaw   = outputs['angle_y_fc'][0][0]
        pitch = outputs['angle_p_fc'][0][0]
        roll  = outputs['angle_r_fc'][0][0]

        return yaw, pitch, roll
