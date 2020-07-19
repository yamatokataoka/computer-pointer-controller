import numpy as np
import logging as log
from model import Model

class Facial_Landmarks_Detection(Model):
    '''
    Class for the Facial Landmarks Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        super(Facial_Landmarks_Detection, self).__init__(model_name, device, extensions)

        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.input_width = self.input_shape[2]
        self.input_height = self.input_shape[3]
        self.output_name = next(iter(self.network.outputs))

    def predict(self, image):
        '''
        This method is meant for running predictions on the input image.
        '''
        input_image = self.preprocess_input(image)

        input_dict={self.input_name:input_image}
        self.exec_network.infer(input_dict)

        outputs = self.exec_network.requests[0].outputs[self.output_name]
        outputs = self.preprocess_output(outputs)

        return outputs

    def preprocess_input(self, image):
        processed_image = super().preprocess_input(image, self.input_height, self.input_width)
        return processed_image

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        preprocess the output.
        '''
        outputs = np.squeeze(outputs)
        eye1_coord = [outputs[0], outputs[1]]
        eye2_coord = [outputs[2], outputs[3]]
        eyes=[eye1_coord, eye2_coord]

        return eyes
