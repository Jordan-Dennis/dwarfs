import equinox
import matplotlib.pyplot as pyplot
import dLux
import typing
import jax.numpy as np
import matplotlib.pyplot as pyplot

apertures = {
        "1": dLux.CircularAperture( 
            x_offset = 1.070652 * np.cos(- np.pi / 4), 
            y_offset = 1.070652 * np.sin(- np.pi / 4),
            radius = 0.078,
            occulting = True,
            softening = True),
        "2": dLux.CircularAperture(
            x_offset = 1.070652 * np.cos(-np.pi / 4 + 2 * np.pi / 3), 
            y_offset = 1.070652 * np.sin(-np.pi / 4 + 2 * np.pi / 3),
            radius = 0.078,
            occulting = True,
            softening = False),
        "3": dLux.CircularAperture(
            x_offset = 1.070652 * np.cos(-np.pi / 4 - 2 * np.pi / 3), 
            y_offset = 1.070652 * np.sin(-np.pi / 4 - 2 * np.pi / 3),
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
            x_offset = - 0.0678822510,
            y_offset = - 0.0678822510,
            radius = 1.2,
            occulting = False,
            softening = True),
        "6": dLux.CircularAperture(
            x_offset = - 0.0678822510,
            y_offset = - 0.0678822510,
            radius = 0.4464,
            occulting = True,
            softening = True),
        "7": dLux.EvenUniformSpider(
            x_offset = - 0.0678822510,
            y_offset = - 0.0678822510,
            number_of_struts = 4,
            width_of_struts = 0.0402,
            rotation = 0.785398163,
            softening = True),
        "8": dLux.SquareAperture(
            x_offset = 1.070652 * np.cos(- np.pi / 4) - 0.0678822510, 
            y_offset = 1.070652 * np.sin(- np.pi / 4) - 0.0678822510,
            theta = - np.pi / 4,
            width = 0.156,
            occulting = True,
            softening = True),
        "9": dLux.SquareAperture(
            x_offset = 1.070652 * np.cos(- np.pi / 4 + 2 * np.pi / 3) - 0.0678822510, 
            y_offset = 1.070652 * np.sin(- np.pi / 4 + 2 * np.pi / 3) - 0.0678822510,
            theta = - np.pi / 4 + np.pi / 3,
            width = 0.156,
            occulting = True, 
            softening = True),
        "10": dLux.SquareAperture(
            x_offset = 1.070652 * np.cos(- np.pi / 4 - 2 * np.pi / 3) - 0.0678822510, 
            y_offset = 1.070652 * np.sin(- np.pi / 4 - 2 * np.pi / 3) - 0.0678822510,
            theta = - np.pi / 3 - np.pi / 4,
            width = 0.156,
            occulting = True,
            softening = True)}

hubble = dLux.OpticalSystem(
    [dLux.CreateWavefront(1024, 2.4), 
     dLux.CompoundAperture(apertures), 
     dLux.PhysicalFFT(57.6)], 
    wavels = [550e-09])

wavefront = dLux.CreateWavefront(1024, 2.4)({"wavelength": 550e-09, "offset": [0., 0.]})["Wavefront"]
pyplot.subplot(2, 1, 1)
pyplot.title("Amplitude")
pyplot.imshow(wavefront.amplitude[0])
pyplot.colorbar()
pyplot.subplot(2, 1, 2)
pyplot.title("Phase")
pyplot.imshow(wavefront.phase[0])
pyplot.colorbar()

pyplot.show()
pyplot.imshow(hubble())
pyplot.colorbar()
pyplot.show()
