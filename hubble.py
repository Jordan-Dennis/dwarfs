import jax 
import tqdm
import optax
import dLux as dl
import equinox as eqx
import jax.numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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


with open("data/filters/HST_NICMOS1.F170M.dat") as filter_data:
    next(filter_data)
    nicmos_filter = np.array([
            [float(entry) for entry in line.strip().split(" ")] 
                for line in filter_data])

nicmos_filter = nicmos_filter\
    .reshape(40, 20, 2)\
    .mean(axis=1)


basis = dl.utils.zernike_basis(5, 256, outside=0.)
target_coeffs = 1e-7 * jax.random.normal(jax.random.PRNGKey(0), [len(basis)])
initial_coeffs = 1e-7 * jax.random.normal(jax.random.PRNGKey(1), [len(basis)])

target_positions = 1e-06 * jax.random.normal(jax.random.PRNGKey(3), (2, 2))
initial_positions = 1e-06 * jax.random.uniform(jax.random.PRNGKey(4), (2, 2))
target_fluxes = 1e7 * jax.random.uniform(jax.random.PRNGKey(5), (2, 1))
initial_fluxes = 1e7 * jax.random.uniform(jax.random.PRNGKey(4), (2, 1))

target_hubble = dl.OpticalSystem(
    [dl.CreateWavefront(256, 2.4, wavefront_type='Angular'),
     dl.TiltWavefront(),
     dl.CompoundAperture({"Hubble": HubblePupil(), 
                          "Nicmos": NicmosColdMask(-0.06788225 / 2., 0.06788225 / 2.)}),
     dl.NormaliseWavefront(),
     dl.ApplyBasisOPD(basis, target_coeffs),
     dl.AngularMFT(dl.utils.arcsec2rad(0.043), 64)], 
    wavels = nicmos_filter[:, 0] * 1e-9, 
    weights = nicmos_filter[:, 1],
    positions = target_positions,
    fluxes = target_fluxes)

np.sqrt(np.sum((target_positions[0] - target_positions[1]) ** 2))

x = target_positions[0][0] - target_positions[1][0]

y = target_positions[0][1] - target_positions[1][1]

180 / np.pi * (np.arctan2(y, x))

target_positions[0]

target_psf = target_hubble.propagate()

plt.imshow(target_psf ** 0.25)
plt.colorbar()

help(jax.random.normal)

vmax

positions

hubble = dl.OpticalSystem(
    [dl.CreateWavefront(256, 2.4, wavefront_type='Angular'), 
     dl.CompoundAperture({"Hubble": HubblePupil(), "Nicmos": NicmosColdMask(0., 0.)}),
     dl.NormaliseWavefront(),
     dl.ApplyBasisOPD(basis, initial_coeffs),
     dl.AngularMFT(dl.utils.arcsec2rad(0.043), 64)], 
    wavels = nicmos_filter[:, 0] * 1e-9, 
    weights = nicmos_filter[:, 1],
    positions = positions,
    fluxes = fluxes)

vmin = -target_hubble.layers[-1].pixel_scale_out * 64 / 2
vmax = -vmin

psf = hubble.propagate()

plt.imshow(psf)
plt.colorbar()
plt.show()

# +
# Define path dict and paths
path_dict = {'zern' : ['layers', 3, 'coeffs'],
             'x'    : ['layers', 1, 'apertures', 'Nicmos', 'delta_x_offset'],
             'y'    : ['layers', 1, 'apertures', 'Nicmos', 'delta_y_offset']}
paths = ['zern', 'x', 'y']

# Update hubble to initialise
hubble = hubble\
    .update_leaves(['zern'], [initial_coeffs], path_dict=path_dict)\
    .update_leaves(['x', 'y'], [0., 0.], path_dict=path_dict)

filter_spec = hubble.get_filter_spec(paths, path_dict=path_dict)


# -

@eqx.filter_jit
@eqx.filter_value_and_grad(arg=filter_spec)
def loss_func(model, target_psf):
    out = model.propagate()
    return np.sum((target_psf - out) ** 2)


# %%timeit
loss, grads = loss_func(hubble, target_psf)

# +
groups = [['x', 'y'], 'zern']
optimisers = [optax.adam(1e-2), optax.adam(1e-7)]
optim = hubble.get_optimiser(groups, optimisers, path_dict=path_dict)
opt_state = optim.init(hubble)

errors, grads_out, models_out = [], [], []

with tqdm.tqdm(range(200), desc='Gradient Descent') as t:
    for i in t: 
        loss, grads = loss_func(hubble, target_psf)
        updates, opt_state = optim.update(grads, opt_state)
        
        current_values = hubble.get_leaves(paths, path_dict=path_dict)
        updated_values = updates.get_leaves(paths, path_dict=path_dict)
        new_values = [current_values[i] + updated_values[i] \
                      for i in range(len(current_values))]
        hubble = hubble.update_leaves(paths, new_values, path_dict=path_dict)
        
        models_out.append(hubble)
        errors.append(loss)
        grads_out.append(grads)

        t.set_description("Loss: {:.3f}".format(loss*1e-3)) #
# -

coordinates = dl.utils.get_pixel_coordinates(256, 2.4 / 256, 0., 0.)

# +
psf = models_out[-1].propagate()
plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["font.family"] = "serif"
plt.rcParams['figure.dpi'] = 120
current_cmap = cm.get_cmap().copy()
current_cmap.set_bad(color="black")
plt.figure(figsize=(12, 9))
plt.subplot(3, 3, 1)
plt.title("Log scale")
plt.imshow(psf ** 0.25)
plt.colorbar()

plt.subplot(3, 3, 2)
plt.title("Log scale")
plt.imshow(target_psf ** 0.25)
plt.colorbar()

plt.subplot(3, 3, 3)
plt.title("Residuals")
plt.imshow((psf - target_psf))
plt.colorbar()

target_aperture = target_hubble\
    .layers[1]\
    ._aperture(coordinates)
plt.subplot(3, 3, 4)
plt.title("Input Aperture")
plt.imshow(target_aperture)
plt.colorbar()

aperture = models_out[-1]\
    .layers[1]\
    ._aperture(coordinates)
plt.subplot(3, 3, 5)
plt.title("Recovered Aperture")
plt.imshow(aperture)
plt.colorbar()

plt.subplot(3, 3, 6)
plt.title("Difference")
plt.imshow(target_aperture - aperture)
plt.colorbar()

target_aberrations = target_hubble\
    .layers[3]\
    .get_total_opd()\
    .at[target_aperture < 0.999]\
    .set(np.nan)
plt.subplot(3, 3, 7)
plt.title("Target Aberrations")
plt.imshow(target_aperture * target_aberrations)
plt.colorbar()

aberrations = models_out[-1]\
    .layers[3]\
    .get_total_opd()\
    .at[aperture < 0.999]\
    .set(np.nan)
plt.subplot(3, 3, 8)
plt.title("Recovered Aberrations")
plt.imshow(aberrations * aperture)
plt.colorbar()

plt.subplot(3, 3, 9)
plt.title("Aberration Residuals")
plt.imshow(target_aberrations - aberrations)
plt.colorbar()
plt.show()
# -

import numpyro

dl.PointSource()



