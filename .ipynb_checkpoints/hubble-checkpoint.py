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
    delta_x_offset: float
    delta_y_offset: float
        
    def __init__(self, x_offset: float, y_offset: float):
        self.delta_x_offset = np.asarray(x_offset).astype(float)
        self.delta_y_offset = np.asarray(y_offset).astype(float)
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
                width_of_struts = 0.0804,
                rotation = 0.785398163,
                softening = True),
            "Mirror Pad 1": dl.SquareAperture(
                x_offset = 1.070652 * np.cos(np.pi / 4) + x_offset, 
                y_offset = 1.070652 * np.sin(np.pi / 4) + y_offset,
                theta = - np.pi / 4,
                width = 0.156,
                occulting = True,
                softening = True),
            "Mirror Pad 2": dl.SquareAperture(
                x_offset = 1.070652 * np.cos(np.pi / 4 + 2 * np.pi / 3) + x_offset, 
                y_offset = 1.070652 * np.sin(np.pi / 4 + 2 * np.pi / 3) + y_offset,
                theta = - np.pi / 4 + np.pi / 3,
                width = 0.156,
                occulting = True, 
                softening = True),
            "Mirror Pad 3": dl.SquareAperture(
                x_offset = 1.070652 * np.cos(np.pi / 4 - 2 * np.pi / 3) + x_offset, 
                y_offset = 1.070652 * np.sin(np.pi / 4 - 2 * np.pi / 3) + y_offset,
                theta = - np.pi / 3 - np.pi / 4,
                width = 0.156,
                occulting = True,
                softening = True)}
    
    
    def set_offset(self, delta_x_offset: float, delta_y_offset: float):
        for aperture in self.apertures:
            self = eqx.tree_at(
                lambda tree: (tree[aperture].x_offset, tree[aperture].y_offset),
                self, 
                (self[aperture].x_offset + delta_x_offset, 
                    self[aperture].y_offset + delta_y_offset))
            
        return eqx.tree_at(
            lambda aperture: (aperture.delta_x_offset, aperture.delta_y_offset), 
                self, (np.asarray(delta_x_offset).astype(float), 
                    np.asarray(delta_y_offset).astype(float)))
    
    def _aperture(self, coordinates: float) -> float:
        self = self.set_offset(self.delta_x_offset, self.delta_y_offset)
        return super()._aperture(coordinates)


class HubblePupil(dl.CompoundAperture):
    def __init__(self):
        self.apertures = {
            "Mirror Pad 1": dl.CircularAperture( 
                x_offset = 1.070652 * np.cos(np.pi / 4), 
                y_offset = 1.070652 * np.sin(np.pi / 4),
                radius = 0.078,
                occulting = True,
                softening = True),
            "Mirror Pad 2": dl.CircularAperture(
                x_offset = 1.070652 * np.cos(np.pi / 4 + 2 * np.pi / 3), 
                y_offset = 1.070652 * np.sin(np.pi / 4 + 2 * np.pi / 3),
                radius = 0.078,
                occulting = True,
                softening = False),
            "Mirror Pad 3": dl.CircularAperture(
                x_offset = 1.070652 * np.cos(np.pi / 4 - 2 * np.pi / 3), 
                y_offset = 1.070652 * np.sin(np.pi / 4 - 2 * np.pi / 3),
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
                width_of_struts = 0.0132 * 2.,
                rotation = 0.785398163,
                softening = True)}


# +
from xaosim.pupil import HST_NIC1 as HST

PSZ = 512   # size of the array for the model
rad = 256
pmask = HST(PSZ, rad, ang=45) # rotated!
# -

dlux_aperture = dl.CompoundAperture({
    "Pupil": HubblePupil(),
    "Nicmos": NicmosColdMask(-0.06788225 / 2., 0.06788225 / 2.)})

coords = dl.utils.get_pixel_coordinates(512, 2.4 / 512, 0., 0.)
aperture = dlux_aperture._aperture(coords)

# +
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(aperture)
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(pmask)
plt.colorbar()
plt.show()
# -

file_name = "data/MAST_2022-08-02T2026/HST/n9nk01010/n9nk01010_mos.fits"

with fits.open(file_name) as hubble_data:
    data = np.array(hubble_data[1].data[46:110, 185:249])

with open("data/filters/HST_NICMOS1.F170M.dat") as filter_data:
    nicmos_filter = np.array(
        [[float(entry) for entry in line.strip().split(" ")] for line in filter_data])


# +
transmissive_dlux_hubble = dl.OpticalSystem(
    [dl.CreateWavefront(512, 2.4, wavefront_type='Angular'), 
     dl.TransmissiveOptic(aperture),
     dl.NormaliseWavefront(),
     dl.AngularMFT(dl.utils.arcsec2rad(0.043), 128)], 
    wavels = nicmos_filter[:, 0] * 1e-9, 
    weights = nicmos_filter[:, 1])

transmissive_xaosim_hubble = dl.OpticalSystem(
    [dl.CreateWavefront(512, 2.4, wavefront_type='Angular'), 
     dl.TransmissiveOptic(pmask),
     dl.NormaliseWavefront(),
     dl.AngularMFT(dl.utils.arcsec2rad(0.043), 64)], 
    wavels = nicmos_filter[:, 0] * 1e-9, 
    weights = nicmos_filter[:, 1])

compound_dlux_hubble = dl.OpticalSystem(
    [dl.CreateWavefront(512, 2.4, wavefront_type='Angular'), 
     dlux_aperture,
     dl.NormaliseWavefront(),
     dl.AngularMFT(dl.utils.arcsec2rad(0.043), 128)], 
    wavels = nicmos_filter[:, 0] * 1e-9, 
    weights = nicmos_filter[:, 1])

# +
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.title("Transmissive dLux")
plt.imshow(transmissive_dlux_hubble.propagate() ** 0.25)
plt.colorbar()

plt.subplot(2, 2, 2)
plt.title("Transmissive xaosim")
plt.imshow(transmissive_xaosim_hubble.propagate() ** 0.25)
plt.colorbar()

plt.subplot(2, 2, 3)
plt.title("Compound dLux")
plt.imshow(compound_dlux_hubble.propagate() ** 0.25)
plt.colorbar()
plt.show()
# -

final_psf, intermediates, layers = hubble.debug_prop((1700) * 1e-9)

fig = plt.figure(figsize=(12, 10))
for i, intermediate in enumerate(intermediates):
    wave = intermediate["Wavefront"].wavefront_to_psf() ** 0.25
    plt.subplot(3, 2, i + 1)
    plt.imshow(wave)
    plt.colorbar()
plt.show()

plt.figure(figsize=(10, 10))
plt.imshow(final_psf ** 0.125,interpolation=None)
plt.colorbar()
plt.show()

data = (data) / data.sum()

# +
psf = hubble.propagate()
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

filter_spec = eqx.tree_at(lambda tree: 
        (tree.layers[1]["Nicmos"].delta_x_offset, 
        tree.layers[1]["Nicmos"].delta_y_offset),
    jax.tree_map(lambda _: False, hubble), (True, True))


@eqx.filter_jit
@eqx.filter_value_and_grad(arg=filter_spec)
def loss_func(model, data):
    out = model.propagate()
    return np.sum((data - out) ** 2)


loss, grads = loss_func(hubble, data)

# +
optim = optax.adam(1e-1)
opt_state = optim.init(hubble)

errors, grads_out, models_out = [], [], []

with tqdm.tqdm(range(5), desc='Gradient Descent') as t:
    for i in t: 
        loss, grads = loss_func(hubble, data)
        updates, opt_state = optim.update(grads, opt_state)
        
        delta_y_offset = updates\
            .layers[1]\
            .apertures["Nicmos"]\
            .delta_y_offset
        
        delta_x_offset = updates\
            .layers[1]\
            .apertures["Nicmos"]\
            .delta_x_offset
        
        nicmos = hubble.layers[1]["Nicmos"]
        
        new_nicmos = nicmos.set_offset(delta_x_offset, delta_y_offset)
        
        hubble = eqx.tree_at(lambda tree: tree.layers[1]["Nicmos"], 
            hubble, new_nicmos)
        
        models_out.append(hubble)
        errors.append(loss)
        grads_out.append(grads)

        t.set_description("Loss: {:.3f}".format(loss)) #

# +
psf = models_out[-1].propagate()
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

aperture = models_out[-1].layers[1]._aperture(dl.utils.get_pixel_coordinates(1024, 0.003, 0., 0.))

plt.imshow(aperture)
plt.colorbar()
plt.show()
