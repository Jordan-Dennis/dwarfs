import equinox
import matplotlib.pyplot as pyplot
import dLux
import typing
import jax.numpy as np

apertures = {
        "1": dLux.SoftEdgedCircularAperture( 
            pixels = 1024,
            pixel_scale = 0.01,
            x_offset = 1.070652, 
            y_offset = 0.0,
            theta = 0.0,
            phi = 0.0,
            magnification = 1.0,
            radius = 0.078),
        "2": dLux.SoftEdgedCircularAperture(
            pixels = 1024,
            pixel_scale = 0.01,
            x_offset = -0.5538,
            y_offset = 0.9066,
            theta = 0.0,
            phi = 0.0,
            magnification = 1.0,
            radius = 0.078),
        "3": dLux.SoftEdgedCircularAperture(
            pixels = 1024,
            pixel_scale = 0.01,
            x_offset = -0.54768,
            y_offset = -0.91272,
            theta = 0.0,
            phi = 0.0,
            magnification = 1.0,
            radius = 0.078),
        "4": dLux.SoftEdgedCircularAperture(
            pixels = 1024,
            pixel_scale = 0.01,
            x_offset = 0.0,
            y_offset = 0.0,
            theta = 0.0,
            phi = 0.0,
            magnification = 1.0,
            radius = 0.396),
        "5": dLux.UniformSpider( 
            number_of_pixels = 1024,
            width_of_image = 1024 * 0.01,
            radius_of_aperture = 1.2,
            number_of_struts = 4,
            width_of_struts = 0.0132,
            rotation = 0.785398163,
            centre_of_aperture = [0.0, 0.0]),
        "6": dLux.SoftEdgedCircularAperture(
            pixels = 1024,
            pixel_scale = 0.01,
            x_offset = -0.096,
            y_offset = 0.0,
            theta = 0.0,
            phi = 0.0,
            magnification = 1.0,
            radius = 0.4464),
        "7": dLux.UniformSpider(
            number_of_pixels = 1024,
            width_of_image = 1024 * 0.01,
            radius_of_aperture = 1.2,
            number_of_struts = 4,
            width_of_struts = 0.0402,
            rotation = 0.785398163,
            centre_of_aperture = [-0.096, 0.0]),
        "8": dLux.SoftEdgedSquareAperture(
            pixels = 1024,
            pixel_scale = 0.01,
            x_offset = -0.7636, 
            y_offset = 0.0,
            theta = 2.12057504,
            phi = 0.0,
            magnification = 1.0,
            width = 0.078),
        "9": dLux.SoftEdgedSquareAperture(
            pixels = 1024,
            pixel_scale = 0.01,
            x_offset = -0.7636, 
            y_offset = 0.0,
            theta = 0.0,
            phi = 0.0,
            magnification = 1.0,
            width = 0.078)}

nicmos = dLux.CompoundAperture(1024, 0.01, apertures)


#for aperture in apertures.values():
#    pyplot.imshow(aperture._aperture())
#    pyplot.colorbar()
#    pyplot.show() 


pyplot.imshow(nicmos._aperture())
pyplot.colorbar()
pyplot.show()
