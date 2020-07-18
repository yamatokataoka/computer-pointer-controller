from model import Model

class Head_Pose_Estimation(Model):
    '''
    Class for the Head Pose Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        super(Head_Pose_Estimation, self).__init__(model_name, device, extensions)

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
        outputs = self.exec_network.infer(input_dict)

        yaw, pitch, roll = self.preprocess_output(outputs)

        return yaw, pitch, roll

    def preprocess_input(self, image):
        processed_image = super().preprocess_input(image, self.input_height, self.input_width)
        return processed_image

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        preprocess the output.
        '''
        yaw   = outputs['angle_y_fc'][0][0]
        pitch = outputs['angle_p_fc'][0][0]
        roll  = outputs['angle_r_fc'][0][0]

        return yaw, pitch, roll
