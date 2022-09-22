import jax 
import tqdm
import optax
import dLux as dl
import equinox as eqx
import jax.numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


class NicmosColdMask(dl.CompoundAperture):
    x_offset: float
    y_offset: float
        
    def __init__(self, x_offset: float, y_offset: float):
        self.x_offset = np.asarray(x_offset).astype(float)
        self.y_offset = np.asarray(y_offset).astype(float)
        self.apertures = {
            "Outer": dl.CircularAperture(
                x_offset = x_offset,
                y_offset = y_offset,
                radius = 1.2,
                occulting = False,
                softening = True),
            "Obstruction": dl.CircularAperture(
                x_offset = x_offset,
                y_offset = y_offset,
                radius = 0.4464,
                occulting = True,
                softening = True),
            "Spider": dl.EvenUniformSpider(
                x_offset = x_offset,
                y_offset = y_offset,
                number_of_struts = 4,
                width_of_struts = 0.0402,
                rotation = 0.785398163,
                softening = True),
            "Mirror Pad 1": dl.SquareAperture(
                x_offset = 1.070652 * np.cos(- np.pi / 4) + x_offset, 
                y_offset = 1.070652 * np.sin(- np.pi / 4) + y_offset,
                theta = - np.pi / 4,
                width = 0.156,
                occulting = True,
                softening = True),
            "Mirror Pad 2": dl.SquareAperture(
                x_offset = 1.070652 * np.cos(- np.pi / 4 + 2 * np.pi / 3) + x_offset, 
                y_offset = 1.070652 * np.sin(- np.pi / 4 + 2 * np.pi / 3) + y_offset,
                theta = - np.pi / 4 + np.pi / 3,
                width = 0.156,
                occulting = True, 
                softening = True),
            "Mirror Pad 3": dl.SquareAperture(
                x_offset = 1.070652 * np.cos(- np.pi / 4 - 2 * np.pi / 3) + x_offset, 
                y_offset = 1.070652 * np.sin(- np.pi / 4 - 2 * np.pi / 3) + y_offset,
                theta = - np.pi / 3 - np.pi / 4,
                width = 0.156,
                occulting = True,
                softening = True)}
        
        
    def set_offset(self, delta_x_offset: float, delta_y_offset: float):
        for aperture in self.apertures:
            self[aperture] = self[aperture]\
                .set_x_offset(self[aperture].get_x_offset() + delta_x_offset)\
                .set_y_offset(self[aperture].get_y_offset() + delta_y_offset)
        
        return eqx.tree_at(
            lambda aperture: (aperture.x_offset, aperture.y_offset), 
                self, (np.asarray(self.x_offset + delta_x_offset).astype(float), 
                    np.asarray(self.y_offset + delta_y_offset).astype(float)))
    

class HubblePupil(dl.CompoundAperture):
    def __init__(self):
        self.apertures = {
            "Mirror Pad 1": dl.CircularAperture( 
                x_offset = 1.070652 * np.cos(- np.pi / 4), 
                y_offset = 1.070652 * np.sin(- np.pi / 4),
                radius = 0.078,
                occulting = True,
                softening = True),
            "Mirror Pad 2": dl.CircularAperture(
                x_offset = 1.070652 * np.cos(-np.pi / 4 + 2 * np.pi / 3), 
                y_offset = 1.070652 * np.sin(-np.pi / 4 + 2 * np.pi / 3),
                radius = 0.078,
                occulting = True,
                softening = False),
            "Mirror Pad 3": dl.CircularAperture(
                x_offset = 1.070652 * np.cos(-np.pi / 4 - 2 * np.pi / 3), 
                y_offset = 1.070652 * np.sin(-np.pi / 4 - 2 * np.pi / 3),
                radius = 0.078,
                occulting = True,
                softening = False),
            "Obstruction": dl.CircularAperture(
                x_offset = 0.0,
                y_offset = 0.0,
                radius = 0.396,
                occulting = True,
                softening = True),
            "Aperture": dl.CircularAperture(
                x_offset = 0.0,
                y_offset = 0.0,
                radius = 1.2,
                occulting = False,
                softening = True),
            "Spider": dl.EvenUniformSpider( 
                x_offset = 0.0,
                y_offset = 0.0,
                number_of_struts = 4,
                width_of_struts = 0.0132,
                rotation = 0.785398163,
                softening = True)}


apertures = {
    "Pupil": HubblePupil(),
    "Nicmos": NicmosColdMask(-0.0678822510,  -0.0678822510)}

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

data = (data) / data.sum()

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

filter_spec = jax.tree_map(lambda _: False, hubble)

filter_spec = eqx.tree_at(
    lambda tree: 
        (tree.layers[1]["Nicmos"].x_offset, tree.layers[1]["Nicmos"].y_offset),
    filter_spec, replace_fn=lambda leaf: leaf is False)


@eqx.filter_jit
@eqx.filter_value_and_grad(arg=filter_spec)
def loss_func(model, data):
    out = model()
    return np.sum((data - out) ** 2)


# +
optim = optax.adam(2.5e1)
opt_state = optim.init(hubble)

errors, grads_out, models_out = [], [], []

with tqdm.tqdm(range(50), desc='Gradient Descent') as t:
    for i in t: 
        loss, grads = loss_func(hubble, data)
        updates, opt_state = optim.update(grads, opt_state)

        model.set_offset()
        
        models_out.append(hubble)
        errors.append(loss)
        grads_out.append(grads)

        t.set_description("Loss: {:.3f}".format(loss)) #
# -

grads

# +
psf = models_out[-1]()
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

errors


