import os
import logging as log
import cv2
from openvino.inference_engine import IENetwork, IECore

class Model:
    '''
    Generic class for model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        set instance variables.
        '''
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.plugin = None
        self.network = None
        self.exec_network = None
        
        self.load_model()

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

        log.debug("IR successfully loaded %s into Inference Engine.", self.model_name)

    def check_model(self):
        '''
        Check adding extensions and any unsupported layers
        '''
        ### Add a CPU extension, if applicable
        if self.extensions and "CPU" in self.device:
            try:
                self.plugin.add_extension(self.extensions, self.device)
            except Exception as e:
                log.error(e)

        ### Check for supported layers
        supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)

        ### Check for any unsupported layers, and let the user
        ### know if anything is missing.
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            log.warning("Unsupported layers found: {}".format(unsupported_layers))
            log.warning("Check whether extensions are available to add to IECore.")

    def preprocess_input(self, image, input_height, input_width):
        '''
        Before feeding the data into the model for inference,
        preprocess image.
        '''
        p_frame = cv2.resize(image, (input_height, input_width))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        return p_frame
