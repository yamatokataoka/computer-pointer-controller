import numpy as np
import logging as log
from model import Model

class Gaze_Estimation(Model):
    '''
    Class for the Gaze Estimation Model.
    '''

    def __init__(self, model_name, device='CPU', extensions=None):
        super(Gaze_Estimation, self).__init__(model_name, device, extensions)

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
