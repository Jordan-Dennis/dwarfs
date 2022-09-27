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


with open("data/filters/HST_NICMOS1.F170M.dat") as filter_data:
    next(filter_data)
    nicmos_filter = np.array(
        [[float(entry) for entry in line.strip().split(" ")] for line in filter_data])




basis = dl.utils.zernike_basis(5, 256, outside=0.)
target_coeffs = 1e-7 * jax.random.normal(jax.random.PRNGKey(0), [len(basis)])
initial_coeffs = 1e-7 * jax.random.normal(jax.random.PRNGKey(0), [len(basis)])

hubble = dl.OpticalSystem(
    [dl.CreateWavefront(256, 2.4, wavefront_type='Angular'), 
     dl.CompoundAperture({"Hubble": HubblePupil(), 
                          "Nicmos": NicmosColdMask(-0.06788225 / 2., 0.06788225 / 2.)}),
     dl.NormaliseWavefront(),
     dl.ApplyBasisOPD(basis, target_coeffs),
     dl.AngularMFT(dl.utils.arcsec2rad(0.043), 64)], 
    wavels = nicmos_filter[:, 0] * 1e-9, 
    weights = nicmos_filter[:, 1])

# +
target_psf = hubble.propagate()
plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1)
plt.title("Log scale")
plt.imshow(target_psf ** 0.25)
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Input Aperture")
plt.imshow(hubble.layers[1]._aperture(dl.utils.get_pixel_coordinates(1024, 0.003, 0., 0.)))
plt.colorbar()
plt.show()
# -

hubble = dl.OpticalSystem(
    [dl.CreateWavefront(256, 2.4, wavefront_type='Angular'), 
     dl.CompoundAperture({"Hubble": HubblePupil(), "Nicmos": NicmosColdMask(0., 0.)}),
     dl.NormaliseWavefront(),
     dl.ApplyBasisOPD(basis, initial_coeffs),
     dl.AngularMFT(dl.utils.arcsec2rad(0.043), 64)], 
    wavels = nicmos_filter[:, 0] * 1e-9, 
    weights = nicmos_filter[:, 1])

filter_spec = eqx.tree_at(lambda tree: 
        (tree.layers[1]["Nicmos"].delta_x_offset, 
        tree.layers[1]["Nicmos"].delta_y_offset,
        tree.layers[3].coeffs),
    jax.tree_map(lambda _: False, hubble), (True, True, True))


@eqx.filter_jit
@eqx.filter_value_and_grad(arg=filter_spec)
def loss_func(model, target_psf):
    out = model.propagate()
    return np.sum((target_psf - out) ** 2)


loss, grads = loss_func(hubble, target_psf)

# +
optim = optax.adam(1e-1)
opt_state = optim.init(hubble)

errors, grads_out, models_out = [], [], []

with tqdm.tqdm(range(10), desc='Gradient Descent') as t:
    for i in t: 
        loss, grads = loss_func(hubble, target_psf)
        updates, opt_state = optim.update(grads, opt_state)
        
        delta_y_offset = updates\
            .layers[1]["Nicmos"]\
            .delta_y_offset
        
        delta_x_offset = updates\
            .layers[1]["Nicmos"]\
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
plt.figure(figsize=(12, 3))
plt.subplot(1, 3, 1)
plt.title("Log scale")
plt.imshow(psf ** 0.25)
plt.colorbar()

plt.subplot(1, 3, 2)
plt.title("Log scale")
plt.imshow(target_psf ** 0.25)
plt.colorbar()

plt.subplot(1, 3, 3)
plt.title("Residuals")
plt.imshow((psf - target_psf))
plt.colorbar()
plt.show()
# -

aperture = models_out[-1].layers[1]._aperture(dl.utils.get_pixel_coordinates(1024, 0.003, 0., 0.))

plt.imshow(aperture)
plt.colorbar()
plt.show()


