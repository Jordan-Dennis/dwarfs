import equinox
import matplotlib.pyplot as pyplot
import dLux
import typing
import jax.numpy as np

apertures = {
        "1": dLux.CircularAperture( 
            x_offset = 1.070652, 
            y_offset = 0.0,
            radius = 0.078,
            occulting = True,
            softening = True),
        "2": dLux.CircularAperture(
            x_offset = -0.5538,
            y_offset = 0.9066,
            radius = 0.078,
            occulting = True,
            softening = False),
        "3": dLux.CircularAperture(
            x_offset = -0.54768,
            y_offset = -0.91272,
            radius = 0.078,
            occulting = True,
            softening = False),
        "4": dLux.CircularAperture(
            x_offset = 0.0,
            y_offset = 0.0,
            radius = 0.396,
            occulting = True,
            softening = True),
        "11": dLux.CircularAperture(
            x_offset = 0.0,
            y_offset = 0.0,
            radius = 1.2,
            occulting = False,
            softening = True),
        "5": dLux.EvenUniformSpider( 
            x_offset = 0.0,
            y_offset = 0.0,
            number_of_struts = 4,
            width_of_struts = 0.0132,
            rotation = 0.785398163,
            softening = True),
        # NOTE: Below this is the cold mask
        "12": dLux.CircularAperture(
            x_offset = 0.096,
            y_offset = 0.0,
            radius = 1.2,
            occulting = False,
            softening = True),
        "6": dLux.CircularAperture(
            x_offset = 0.096,
            y_offset = 0.0,
            radius = 0.4464,
            occulting = True,
            softening = True),
        "7": dLux.EvenUniformSpider(
            x_offset = -0.096,  
            y_offset = 0.0,
            number_of_struts = 4,
            width_of_struts = 0.0402,
            rotation = 0.785398163,
            softening = True),
        "8": dLux.SquareAperture(
            x_offset = -0.478775884, 
            y_offset = -0.781291236,
            theta = -2.12057504,
            width = 0.078,
            occulting = True,
            softening = True),
        "9": dLux.SquareAperture(
            x_offset = 0.91632, 
            y_offset = 0.0,
            theta = 0.0,
            width = 0.078,
            occulting = True, 
            softening = True),
        "10": dLux.SquareAperture(
            x_offset = -0.471396840, 
            y_offset = 0.785559492,
            theta = 2.11184839,
            width = 0.078,
            occulting = True,
            softening = True)}

nicmos = dLux.CompoundAperture(apertures)

coordinates = dLux.utils.get_pixel_coordinates(1024, 0.003, 0., 0.)

for i, aperture in enumerate(apertures.values()):
    pyplot.subplot(3, 4, i + 1)
    pyplot.imshow(aperture._aperture(coordinates))
    pyplot.colorbar()
pyplot.show() 

pyplot.imshow(nicmos._aperture(coordinates))
pyplot.colorbar()
pyplot.show()
