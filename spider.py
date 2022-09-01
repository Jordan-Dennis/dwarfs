# TODO: This is going to be a script for generating arbitrary support 
# spiders at arbitrary rotation. 
#class Spider():
#    npix
#    radius_of_spider
#    width_of_image
#
#    def _sigmoid(width):
#        jax.nn.sigmoid
#
#    def _strut(angle, width):
#
#    @abstractmethod
#    def _spider():
#
#    def __call__():
#        soft_edged_spider = self.sigmoid(self._spider())
#  
#class UniformSpider():
#    number
#    rotation
#    width
#
#    def __init__():
#    
#    def _spider():

class Spider(equinox.Module, abc.ABC):
    number_of_pixel: int
    radius_of_spider: float
    width_of_image: float
    centre_of_spider: (float, float) 

    def _strut(self: Layer, angle: float) -> Array:
        get_pixel_positions(number_of_pixels, 0., 0.)


    def _sigmoid(self: Layer, structure: Array) -> Array:
        return 


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
