from model import Model

class Face_Detection(Model):
    '''
    Class for the Face Detection Model.
    '''

    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.6):
        super(Face_Detection, self).__init__(model_name, device, extensions)

        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.input_width = self.input_shape[2]
        self.input_hight = self.input_shape[3]
        self.output_name = next(iter(self.network.outputs))

        self.threshold = threshold

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
        super().preprocess_input(image, self.input_hight, self.input_width)

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        preprocess the output.
        '''
        outputs = outputs[0][0]
        outputs[outputs[:, 2] >= self.threshold]

        return outputs
