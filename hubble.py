import dLux as dl
import jax.numpy as np
import matplotlib.pyplot as plt

apertures = {
        "1": dl.CircularAperture( 
            x_offset = 1.070652 * np.cos(- np.pi / 4), 
            y_offset = 1.070652 * np.sin(- np.pi / 4),
            radius = 0.078,
            occulting = True,
            softening = True),
        "2": dl.CircularAperture(
            x_offset = 1.070652 * np.cos(-np.pi / 4 + 2 * np.pi / 3), 
            y_offset = 1.070652 * np.sin(-np.pi / 4 + 2 * np.pi / 3),
            radius = 0.078,
            occulting = True,
            softening = False),
        "3": dl.CircularAperture(
            x_offset = 1.070652 * np.cos(-np.pi / 4 - 2 * np.pi / 3), 
            y_offset = 1.070652 * np.sin(-np.pi / 4 - 2 * np.pi / 3),
            radius = 0.078,
            occulting = True,
            softening = False),
        "4": dl.CircularAperture(
            x_offset = 0.0,
            y_offset = 0.0,
            radius = 0.396,
            occulting = True,
            softening = True),
        "11": dl.CircularAperture(
            x_offset = 0.0,
            y_offset = 0.0,
            radius = 1.2,
            occulting = False,
            softening = True),
        "5": dl.EvenUniformSpider( 
            x_offset = 0.0,
            y_offset = 0.0,
            number_of_struts = 4,
            width_of_struts = 0.0132,
            rotation = 0.785398163,
            softening = True),
        # NOTE: Below this is the cold mask
        "12": dl.CircularAperture(
            x_offset = - 0.0678822510,
            y_offset = - 0.0678822510,
            radius = 1.2,
            occulting = False,
            softening = True),
        "6": dl.CircularAperture(
            x_offset = - 0.0678822510,
            y_offset = - 0.0678822510,
            radius = 0.4464,
            occulting = True,
            softening = True),
        "7": dl.EvenUniformSpider(
            x_offset = - 0.0678822510,
            y_offset = - 0.0678822510,
            number_of_struts = 4,
            width_of_struts = 0.0402,
            rotation = 0.785398163,
            softening = True),
        "8": dl.SquareAperture(
            x_offset = 1.070652 * np.cos(- np.pi / 4) - 0.0678822510, 
            y_offset = 1.070652 * np.sin(- np.pi / 4) - 0.0678822510,
            theta = - np.pi / 4,
            width = 0.156,
            occulting = True,
            softening = True),
        "9": dl.SquareAperture(
            x_offset = 1.070652 * np.cos(- np.pi / 4 + 2 * np.pi / 3) - 0.0678822510, 
            y_offset = 1.070652 * np.sin(- np.pi / 4 + 2 * np.pi / 3) - 0.0678822510,
            theta = - np.pi / 4 + np.pi / 3,
            width = 0.156,
            occulting = True, 
            softening = True),
        "10": dl.SquareAperture(
            x_offset = 1.070652 * np.cos(- np.pi / 4 - 2 * np.pi / 3) - 0.0678822510, 
            y_offset = 1.070652 * np.sin(- np.pi / 4 - 2 * np.pi / 3) - 0.0678822510,
            theta = - np.pi / 3 - np.pi / 4,
            width = 0.156,
            occulting = True,
            softening = True)}

hubble = dl.OpticalSystem(
    [dl.CreateWavefront(512, 2.4, wavefront_type='Angular'), 
     dl.CompoundAperture(apertures), 
     dl.NormaliseWavefront(),
     dl.AngularMFT(dl.utils.arcsec2rad(0.01), 256)], 
    wavels = [550e-09])

psf = hubble()
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Linear scale")
plt.imshow(psf)
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Log scale")
plt.imshow(np.log10(psf))
plt.colorbar()
plt.show()
from astropy.io import fits
from matplotlib import pyplot

file_name = "data/MAST_2022-08-02T2026/HST/n9nk01010/n9nk01010_mos.fits"

with fits.open(file_name) as hubble_data:
    data = hubble_data[1].data

pyplot.imshow(data)
pyplot.colorbar()
pyplot.show()
