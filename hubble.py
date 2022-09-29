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
initial_positions = jax.numpy.array([[0., 0.], [0., 0.]])
target_fluxes = 1e7 * jax.random.uniform(jax.random.PRNGKey(5), (2, 1))
initial_fluxes = 1e7 * jax.numpy.array([1., 1.])

x_offset = -0.06788225 / 2.
y_offset = 0.06788225 / 2.

target_hubble = dl.OpticalSystem(
    [dl.CreateWavefront(256, 2.4, wavefront_type='Angular'),
     dl.TiltWavefront(),
     dl.CompoundAperture({"Hubble": HubblePupil(), 
                          "Nicmos": NicmosColdMask(x_offset, y_offset)}),
     dl.NormaliseWavefront(),
     dl.ApplyBasisOPD(basis, target_coeffs),
     dl.AngularMFT(dl.utils.arcsec2rad(0.043), 64)], 
    wavels = nicmos_filter[:, 0] * 1e-9, 
    weights = nicmos_filter[:, 1],
    positions = target_positions,
    fluxes = target_fluxes)

# +
hubble = dl.OpticalSystem(
    [dl.CreateWavefront(256, 2.4, wavefront_type='Angular'), 
     dl.TiltWavefront(),
     dl.CompoundAperture({"Hubble": HubblePupil(), "Nicmos": NicmosColdMask(0., 0.)}),
     dl.NormaliseWavefront(),
     dl.ApplyBasisOPD(basis, initial_coeffs),
     dl.AngularMFT(dl.utils.arcsec2rad(0.043), 64)], 
    wavels = nicmos_filter[:, 0] * 1e-9, 
    weights = nicmos_filter[:, 1],
    positions = initial_positions,
    fluxes = initial_fluxes)

# Define path dict and paths
path_dict = {'zern': ['layers', -2, 'coeffs'],
             'x_offset': ['layers', 2, 'apertures', 'Nicmos', 'delta_x_offset'],
             'y_offset': ['layers', 2, 'apertures', 'Nicmos', 'delta_y_offset'],
             'positions': ['positions'],
             'fluxes': ['fluxes']}
paths = ['zern', 'x_offset', 'y_offset', 'positions', 'fluxes']

filter_spec = hubble.get_filter_spec(paths, path_dict=path_dict)

@eqx.filter_jit
@eqx.filter_value_and_grad(arg=filter_spec)
def loss_func(model, target_psf):
    out = model.propagate()
    return np.sum((target_psf - out) ** 2)

loss, grads = loss_func(hubble, target_psf)

groups = [['x_offset', 'y_offset'], 'zern', 'positions', 'fluxes']
optimisers = [optax.adam(1e-12), optax.adam(1e-12), optax.adam(1e-6), optax.adam(1e-12)]
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
plt.imshow(target_psf ** 0.25)
plt.colorbar()

plt.subplot(3, 3, 2)
plt.title("Log scale")
plt.imshow(psf ** 0.25)
plt.colorbar()

plt.subplot(3, 3, 3)
plt.title("Residuals")
plt.imshow((psf - target_psf))
plt.colorbar()

target_aperture = target_hubble\
    .layers[2]\
    ._aperture(coordinates)
plt.subplot(3, 3, 4)
plt.title("Input Aperture")
plt.imshow(target_aperture)
plt.colorbar()

aperture = models_out[-1]\
    .layers[2]\
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
    .layers[4]\
    .get_total_opd()\
    .at[target_aperture < 0.999]\
    .set(np.nan)
plt.subplot(3, 3, 7)
plt.title("Target Aberrations")
plt.imshow(target_aperture * target_aberrations)
plt.colorbar()

aberrations = models_out[-1]\
    .layers[4]\
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
plt.savefig("../report_plan/binary_gradient_descent.pdf")
plt.show()
# -

x_resid = x_offset - np.array([m.get_leaf('x_offset', path_dict=path_dict) 
                               for m in models_out])
y_resid = y_offset - np.array([m.get_leaf('y_offset', path_dict=path_dict) 
                               for m in models_out])
pos_resid = target_positions - np.array([m.get_leaf("positions", path_dict=path_dict) 
                                         for m in models_out])
coeff_resid = target_coeffs - np.array([m.get_leaf("zern", path_dict=path_dict) 
                                        for m in models_out])

# +
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.title("x offset")
plt.plot(x_resid)
plt.plot(y_resid)

plt.subplot(1, 3, 2)
plt.title("Position")
plt.plot(pos_resid[:, 0, 0])
plt.plot(pos_resid[:, 0, 1])
plt.plot(pos_resid[:, 1, 0])
plt.plot(pos_resid[:, 1, 1])

plt.subplot(1, 3, 3)
plt.title("Coeffs")
plt.plot(coeff_resid[:, 0])
plt.plot(coeff_resid[:, 1])
plt.plot(coeff_resid[:, 2])
plt.plot(coeff_resid[:, 3])
plt.plot(coeff_resid[:, 4])
plt.show()
# -

import numpyro


def binary_model(target_psf)
