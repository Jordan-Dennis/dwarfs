import jax 
import dLux as dl
import jax.numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

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

file_name = "data/MAST_2022-08-02T2026/HST/n9nk01010/n9nk01010_mos.fits"

with fits.open(file_name) as hubble_data:
    data = np.array(hubble_data[1].data[46:110, 185:249])

with open("data/filters/HST_NICMOS1.F170M.dat") as filter_data:
    nicmos_filter = np.array(
        [[float(entry) for entry in line.strip().split(" ")] for line in filter_data])


plt.plot(nicmos_filter[:, 0] * 1e-9, nicmos_filter[:, 1])
plt.show()

hubble = dl.OpticalSystem(
    [dl.CreateWavefront(512, 2.4, wavefront_type='Angular'), 
     dl.CompoundAperture(apertures), 
     dl.NormaliseWavefront(),
     dl.AngularMFT(dl.utils.arcsec2rad(0.043), 64)], 
    wavels = nicmos_filter[:, 0] * 1e-9, 
    weights = nicmos_filter[:, 1])

# +
psf = hubble()
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.title("Linear scale")
plt.imshow(psf)
plt.colorbar()

plt.subplot(2, 2, 2)
plt.title("Log scale")
plt.imshow(psf ** 0.25)
plt.colorbar()

plt.subplot(2, 2, 3)
plt.title("Linear scale")
plt.imshow(data)
plt.colorbar()

plt.subplot(2, 2, 4)
plt.title("Log scale")
plt.imshow(data ** 0.25)
plt.colorbar()
plt.show()
# -

filter_spec = model.get_filter_spec([coeffs_path])
@eqx.filter_jit
@eqx.filter_value_and_grad(arg=filter_spec)
def loss_func(model, data):
    out = model.propagate()
    return -np.sum(jax.scipy.stats.poisson.logpmf(data, out))
