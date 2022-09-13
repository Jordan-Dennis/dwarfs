import abc 
import jax
import typing
import equinox
import functools
import dLux.utils 
import numpy as np
#import jax.numpy as np 
import matplotlib.pyplot as pyplot


Layer = typing.TypeVar("Layer")


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
    width_of_image: float
    number_of_pixels: int
    radius_of_spider: float
    centre_of_spider: float


    def __init__(
            self: Layer, 
            width_of_image: float,
            number_of_pixels: int, 
            radius_of_spider: float,
            centre_of_spider: float) -> Layer:
        self.number_of_pixels = number_of_pixels
        self.width_of_image = np.asarray(width_of_image).astype(float)
        self.centre_of_spider = np.asarray(centre_of_spider).astype(float)
        self.radius_of_spider = np.asarray(radius_of_spider).astype(float)


    def _coordinates(self: Layer) -> float:
        pixel_scale = self.width_of_image / self.number_of_pixels
        pixel_centre = self.centre_of_spider / pixel_scale
        pixel_coordinates = dLux.utils.get_pixel_positions(
            self.number_of_pixels, pixel_centre[0], pixel_centre[1])
        return pixel_coordinates * pixel_scale  
 

    def _rotate(self: Layer, image: float, angle: float) -> float:
        coordinates = self._coordinates()
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)], 
            [np.sin(angle), np.cos(angle)]])
        return np.apply_along_axis(np.matmul, 0, coordinates, rotation_matrix) 


    # TODO: This needs to be truncated to the radius of the spider. 
    # @functools.partial(jax.vmap, in_axes=(None, 0))
    def _strut(self: Layer, angle: float) -> float:
        coordinates = self._rotate(self._coordinates(), angle)
        distance = np.abs(coordinates[1] * (coordinates[0] > 0))
        spider = self._sigmoid(distance)

        radial_coordinates = np.hypot(coordinates[0], coordinates[1])
        radial_distance = np.abs(radial_coordinates - self.radius_of_spider)
        
        return spider * radial_distance
        

    def _sigmoid(self: Layer, distance: float) -> float:
        steepness = 10
        return np.tanh(steepness * distance)


    @abc.abstractmethod
    def _spider(self: Layer) -> float:
        pass 

    
    @abc.abstractmethod
    def __call__(self: Layer, params: dict) -> dict:
        pass 


class UniformSpider(Spider):
    number_of_struts: int
    width_of_struts: float
    rotation: float


    def __init__(
            self: Layer, 
            width_of_image: float,
            number_of_pixels: int, 
            radius_of_spider: float,
            centre_of_spider: float,
            number_of_struts: int, 
            width_of_struts: float, 
            rotation: float) -> Layer:
        super().__init__(
            width_of_image, 
            number_of_pixels, 
            radius_of_spider,
            centre_of_spider)
        self.number_of_struts = number_of_struts
        self.rotation = np.asarray(rotation).astype(float)
        self.width_of_struts = np.asarray(width_of_struts).astype(float)


    def _spider(self: Layer) -> float:
        angles = np.linspace(0, 2 * np.pi, self.number_of_struts, 
            endpoint=False)
        angles += self.rotation
        spider = np.zeros((self.number_of_pixels, self.number_of_pixels))
        for angle in angles:
            spider += self._strut(angle)
        return spider / self.number_of_struts
        # return self._strut(angles).sum(axis=0) / self.number_of_struts
        
 
    def __call__(self: Layer, params: dict) -> dict:
        aperture = self._spider()
        wavefront = params["Wavefront"]
        wavefront = wavefront\
            .set_amplitude(wavefront.get_amplitude() * aperture)\
            .set_phase(wavefront.get_phase() * aperture)
        params["Wavefront"] = wavefront
        return params

spider = UniformSpider(1., 1024, .5, [0., 0.], 4, 0.01, 0.)._spider()
pyplot.imshow(spider)
pyplot.colorbar()
pyplot.show()
