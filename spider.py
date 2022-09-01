# TODO: This is going to be a script for generating arbitrary support 
# spiders at arbitrary rotation. 


class Spider(equinox.Module, abc.ABC):
    """
    An abstraction on the concept of an optical spider for a space telescope.
    These are the things that hold up the secondary mirrors. For example,

        ###########
        ###  #   ##
        #    #    #
        ###########
        #    #    #
        ##   #   ##
        ###########

    Is a shody representation of a 4 structured spider like the Hubble Space 
    Telescope spider. 

    Parameters
    ----------
    number_of_pixels: int
    radius_of_spider: float
    width_of_image: float
    center_of_spicer: Array
    """
        
    number_of_pixel: int
    radius_of_spider: float
    width_of_image: float
    centre_of_spider: Array


    def _coordinates(self: Layer) -> Array:
        pixel_scale = self.width_of_image / self.number_of_pixels
        pixel_centre = centre_of_spider / pixel_scale
        pixel_coordinates = get_pixel_positions(number_of_pixels, 
            pixel_centre[0], pixel_centre[1])
        return pixel_coordinates * pixel_scale  
 


    # TODO: This needs to be truncated to the radius of the spider. 
    def _strut(self: Layer, angle: float) -> Array:
        coordinates = self._coordinates()
        distance = np.abs(coordinates[0] * np.cos(angle) \
            - coordinates[1] / np.sin(angle))
        return distance
        

    def _sigmoid(self: Layer, angle: float, width: float) -> Array:
        steepness = 10
        distance = self._struct(angle)
        return np.tanh(steepness * (distance - width))


    @abstractmethod
    def _spider(self: Layer) ->  Array:
        pass 

    
    @abstractmethod
    def __call__(self: Layer, params: dict) -> dict:
        pass 


class UniformSpider(equinox.Module, abc.ABC):
    number_of_struts: int
    width_of_struts: float
    rotation: float


    def __init__(
            self: Layer, 
            number_of_struts: int, 
            width_of_struts: float, 
            rotation: float) -> Layer:
        self.number_of_struts = number_of_struts
        self.width_of_struts = width_of_struts
        self.rotation = rotation


    def _spider()


    def __call__(self: Layer, params: dict) -> dict:
