import jax
import jax.numpy as np
import jax.random as jr
import numpyro as npy
import numpyro.distributions as dist
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=2'

import equinox as eqx
import optax
import dLux as dl
import matplotlib.pyplot as plt
import chainconsumer as cc
import jupyterthemes.jtplot as jtplot

# %matplotlib inline
plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["font.family"] = 'serif'
plt.rcParams["text.usetex"] = 'true'
plt.rcParams['figure.dpi'] = 120

class NicmosColdMask(dl.CompoundAperture):
    x_offset: float
    y_offset: float
    mirror_pad_radius: float
    mirror_pad_angles: float
    relative_position: dict
        
    def __init__(self, x_offset: float, y_offset: float):
        self.x_offset = np.asarray(x_offset).astype(float)
        self.y_offset = np.asarray(y_offset).astype(float)
        self.mirror_pad_radius = np.asarray(1.070652).astype(float)
        self.mirror_pad_angles = np.array([-2*np.pi/3, 0., 2*np.pi/3]) + np.pi/4
        self.relative_position = {
            "Outer": (0., 0.),
            "Obstruction": (0., 0.),
            "Spider": (0., 0.),
            "Mirror Pad 1": (self.mirror_pad_radius*np.cos(self.mirror_pad_angles[0]),
                             self.mirror_pad_radius*np.sin(self.mirror_pad_angles[0])),
            "Mirror Pad 2": (self.mirror_pad_radius*np.cos(self.mirror_pad_angles[1]),
                             self.mirror_pad_radius*np.sin(self.mirror_pad_angles[1])),
            "Mirror Pad 3": (self.mirror_pad_radius*np.cos(self.mirror_pad_angles[2]),
                             self.mirror_pad_radius*np.sin(self.mirror_pad_angles[2]))
        }
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
    
    
    def set_offset(self, x_offset: float, y_offset: float):
        x_offset = np.asarray(x_offset).astype(float)
        y_offset = np.asarray(y_offset).astype(float)
        
        for aperture in self.apertures:
            new_x_offset = self.relative_position[aperture][0] + x_offset
            new_y_offset = self.relative_position[aperture][1] + y_offset
            new_aperture = self\
                .apertures[aperture]\
                .set_x_offset(new_x_offset)\
                .set_y_offset(new_y_offset)
            self.apertures[aperture] = new_aperture          
            
        return eqx.tree_at(
            lambda aperture: (aperture.x_offset, aperture.y_offset), 
            self, (x_offset, y_offset))
    
    def _aperture(self, coordinates: float) -> float:
        # print(f"x_offset: {self.delta_x_offset}")
        # print(f"y_offset: {self.delta_y_offset}")
        updated_self = self.set_offset(self.x_offset, self.y_offset)
        return super(NicmosColdMask, updated_self)._aperture(coordinates)


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

with open("HST_NICMOS1.F170M.dat") as filter_data:
    next(filter_data)
    nicmos_filter = np.array([
            [float(entry) for entry in line.strip().split(" ")] 
                for line in filter_data])
    
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.title("Raw filter")
plt.scatter(nicmos_filter[:, 0], nicmos_filter[:, 1])

nicmos_filter = nicmos_filter\
    .reshape(10, 80, 2)\
    .mean(axis=1)\
    .at[:, 0]\
    .mul(1e-9)

# Nicmos filter is defined in percentages, not throughput, so divide by 100
nicmos_filter = nicmos_filter.at[:, 1].divide(100)
dl_nicmos_filter = dl.Filter(nicmos_filter[:, 0], nicmos_filter[:, 1])

plt.subplot(1, 2, 2)
plt.title("Input filter")
plt.scatter(dl_nicmos_filter.wavelengths, dl_nicmos_filter.throughput)
plt.show()

total_throughput = dl_nicmos_filter.throughput.sum()/len(dl_nicmos_filter.throughput)
print("Total filter throughput: {}%".format(total_throughput*100))

# wavelengths = np.tile(nicmos_filter[:, 0], (2, 1))
wavelengths = np.tile(1e-6*np.linspace(1.6, 1.8, 3), (2, 1))
# weights = np.ones((2,) + nicmos_filter[:, 0].shape)
weights = np.ones(wavelengths.shape)
combined_spectrum = dl.CombinedSpectrum(wavelengths, weights).normalise()

# Create Binary Source,
true_position = np.zeros(2)
true_separation, true_field_angle = dl.utils.arcsec2rad(5e-1), 0
true_flux, true_flux_ratio = 1e5, 2
resolved = [False, False]
binary_source = dl.BinarySource(true_position, true_flux, true_separation, 
                             true_field_angle, true_flux_ratio, 
                             combined_spectrum, resolved, name="Binary")

# Construct Optical system
wf_npix = 128
det_npix = 64

# Zernike aberrations,
basis = dl.utils.zernike_basis(6, npix=wf_npix)[3:] * 1e-9
true_coeffs = jr.normal(jr.PRNGKey(0), (basis.shape[0],))

true_x_offset, true_y_offset = 0.067, -0.067
pupils = {"Hubble": HubblePupil(), "Nicmos": NicmosColdMask(true_x_offset, true_y_offset)}

# Construct optical layers,
true_pixel_scale = dl.utils.arcsec2rad(0.043)
layers = [dl.CreateWavefront(wf_npix, 2.4, wavefront_type="Angular"),
          dl.TiltWavefront(),
          dl.CompoundAperture(pupils),
          dl.ApplyBasisOPD(basis, true_coeffs),
          dl.NormaliseWavefront(),
          dl.AngularMFT(true_pixel_scale, det_npix)]

# Construct Detector,
true_bg = 10.
true_pixel_response = 1 + 0.05*jr.normal(jr.PRNGKey(0), (det_npix, det_npix))
detector_layers = [
    dl.AddConstant(true_bg),
    # dl.ApplyPixelResponse(true_pixel_response),

# Construct Telescope,
telescope = dl.Telescope(dl.Optics(layers), 
                         dl.Scene([binary_source]),
                         filter=dl_nicmos_filter,
                         detector=dl.Detector(detector_layers))

## Gerenate psf,
psf = telescope.model_scene()
psf_photon = jr.poisson(jr.PRNGKey(0), psf)
bg_noise = true_bg + jr.normal(jr.PRNGKey(0), psf_photon.shape)
image = psf_photon + bg_noise
data = image.flatten()

plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.title("PSF")
plt.imshow(psf ** 0.25)
plt.colorbar()

plt.subplot(1, 3, 2),
plt.title("PSF + Photon")
plt.imshow(psf_photon ** 0.25)
plt.colorbar()

plt.subplot(1, 3, 3)
plt.title("Data")
plt.imshow(image ** 0.25)
plt.colorbar()
plt.show()

# Lets define our path dict to simplify accessing these attributes
# These can all always be defined, there is no need to comment them out
path_dict = {
    'pos'      : ['scene',    'sources', 'Binary',             'position'       ],
    'sep'      : ['scene',    'sources', 'Binary',             'separation'     ],
    'angle'    : ['scene',    'sources', 'Binary',             'field_angle'    ],
    'flx'      : ['scene',    'sources', 'Binary',             'flux'           ],
    'cont'     : ['scene',    'sources', 'Binary',             'flux_ratio'     ],
    'zern'     : ['optics',   'layers',  'Apply Basis OPD',    'coeffs'         ],
    'bg'       : ['detector', 'layers',  'AddConstant',        'value'          ],
    'FF'       : ['detector', 'layers',  'ApplyPixelResponse', 'pixel_response' ],
    'xoffset' : ['optics',   'layers',  'CompoundAperture', 'apertures', 'Nicmos', 'x_offset'],
    'yoffset' : ['optics',   'layers',  'CompoundAperture', 'apertures', 'Nicmos', 'y_offset']
}

# telescope.get_leaves(['yoffset'], path_dict=path_dict)
# telescope.get_leaves([['optics',   'layers',  'CompoundAperture', 'apertures', 'Nicmos']], path_dict=path_dict)[0]

telescope.scene.sources['Binary'].field_angle

np.arctan2(-0.1, 1)

def psf_model(data, model, path_dict=None):
    # Define empty paths and values lists to append to,
    paths, values = [], []
    
    # # Position
    position_pix = npy.sample("position_pix", dist.Uniform(-4, 4), sample_shape=(2,))
    position     = npy.deterministic('position', position_pix * true_pixel_scale)
    paths.append('pos'), values.append(position)
    
    # Separation
    log_sep_min = np.log(true_separation - 2 * true_pixel_scale)
    log_sep_max = np.log(true_separation + 2 * true_pixel_scale)
    separation_log = npy.sample("log_sep", dist.Uniform(log_sep_min, log_sep_max))
    separation     = npy.deterministic('separation', np.exp(separation_log))
    paths.append('sep'), values.append(separation)
    
    # Field Angle (Position Angle),
    # theta_x = npy.sample("theta_x", dist.Normal(0, 1))
    # theta_y = npy.sample("theta_y", dist.Normal(0, 1))
    # field_angle = npy.deterministic('field_angle', np.arctan2(theta_y, theta_x))
    field_angle = npy.sample("field_angle", dist.Uniform(-np.pi/4, np.pi/4))
    paths.append('angle'), values.append(field_angle)
    
    # Flux,
    flux_log = npy.sample('log_flux', dist.Uniform(4, 8))
    flux     = npy.deterministic('flux', 10**flux_log)
    paths.append('flx'), values.append(flux)
    
    # Flux ratio,
    flux_ratio_log = npy.sample('log_flux_ratio', dist.Uniform(0, 4))
    flux_ratio     = npy.deterministic('flux_ratio', 10**flux_ratio_log)
    paths.append('cont'), values.append(flux_ratio)
    
    # Zernikes
    coeffs = npy.sample("coeffs", dist.Normal(0, 2), sample_shape=true_coeffs.shape)
    paths.append('zern'), values.append(coeffs)
    
    # Background
    bg = npy.sample("bg", dist.Uniform(5, 15))
    paths.append('bg'), values.append(bg)
    
    # Offset 
    x_offset_latent = npy.sample("offset_x_latent", dist.HalfNormal(1))
    y_offset_latent = npy.sample("offset_y_latent", dist.HalfNormal(1))
    x_offset = npy.deterministic('offset_x', 0.1*x_offset_latent)
    y_offset = npy.deterministic('offset_y', -0.1*y_offset_latent)
    paths.append('xoffset'), values.append(x_offset)
    paths.append('yoffset'), values.append(y_offset)

    
    with npy.plate("data", len(data)):
        poisson_model = dist.Poisson(model.update_and_model(
            "model_image", paths, values, path_dict=path_dict, flatten=True))
    
    return npy.sample("psf", poisson_model, obs=data)

telescope.optics.layers['CompoundAperture']['Nicmos'].x_offset, telescope.optics.layers['CompoundAperture']['Nicmos'].y_offset

sampler = npy.infer.MCMC(
    npy.infer.NUTS(psf_model),    
    num_warmup=1000,
    num_samples=1000,
    num_chains=jax.device_count(),
    progress_bar=True)

sampler.run(jr.PRNGKey(0), data, telescope, path_dict=path_dict)

values_out = sampler.get_samples()
sampler.print_summary()

def make_dict(dict_in, truth=False):
    znames = ['Focus', 'Astig45', 'Astig0', 'ComaY', 'ComaX', 'TfoilY', 'TfoilX']
    pos_names = ['Pos_x', 'Pos_y']
    name_dict = {'separation': 'r', 
                 'field_angle': r'$\\phi$',
                 'flux_ratio': 'Contrast', 
                 'flux':  r'$\\overline{flux}$',
                 'bg': r'$\\mu_{BG}$', 
                 'bg_var': r'$\\sigma_{BG}$',
                 'pixel_scale':  'pixscale',
                 'offset_x': 'offset_x',
                 'offset_y': 'offset_y'}
    
    dict_out = {}
    keys = list(dict_in.keys())
    for i in range(len(keys)):
        key = keys[i]
        # if 'latent' in key or 'log' in key or 'theta' in key or '_pix' in key or '_raw' in key:# or key == 'bg':,
        if 'latent' in key or 'log' in key or '_pix' in key or '_raw' in key:# or key == 'bg':,
            continue
        item = dict_in[key]
        if key == 'position':
            for j in range(item.shape[-1]):
                dict_out[pos_names[j]] = item[j] if truth else item[:, j]
                    
        elif key == 'coeffs':
            for j in range(item.shape[-1]):
                dict_out[znames[j]] = item[j] if truth else item[:, j]
        else:
            dict_out[name_dict[key]] = item

    # print(list(dict_out.keys()))
    # Now re-order for nicer plotting
    order = [
        'r', 
        r'$\\phi$', 
        'Pos_x', 
        'Pos_y', 
        r'$\\overline{flux}$', 
        'Contrast', 
        r'$\\mu_{BG}$', 
        'offset_x',
        'offset_y',
        'Focus', 
        'Astig45', 
        'Astig0'
    ]
    
    new_dict = {}
    for key in order:
        new_dict[key] = dict_out[key]
    return new_dict

# Format chains for plotting
# This can always stay defined, no need to comment out
truth_dict = {
    'bg':          true_bg,          
    'coeffs':   true_coeffs, 
    'field_angle': true_field_angle, 
    'flux':     true_flux, 
    'flux_ratio':  true_flux_ratio,  
    'position': true_position, 
    'separation':  true_separation,  
    'bg_var':   1.,
    'pixel_scale': true_pixel_scale,
    'offset_x': 0.067,
    'offset_y': -0.067,
}

chain = cc.ChainConsumer()
chain.add_chain(chain_dict)
chain.configure(serif=True, shade=True, bar_shade=True, 
                shade_alpha=0.2, spacing=1., max_ticks=3)
fig = chain.plotter.plot(truth=truth_dict_in)
# fig.set_size_inches((4,4))
fig.set_size_inches((12,12))
fig.savefig('hmc', dpi=200, facecolor='w')

est_field_angle = values_out['field_angle'].mean()

binary_source = dl.BinarySource(true_position, true_flux, true_separation, 
                             est_field_angle, true_flux_ratio, 
                             combined_spectrum, resolved, name="Binary")

# Construct Telescope,
telescope_est = dl.Telescope(dl.Optics(layers), 
                         dl.Scene([binary_source]),
                         filter=dl_nicmos_filter,
                         detector=dl.Detector(detector_layers))

# +
## Gerenate psf,
psf = telescope_est.model_scene()
psf_photon = jr.poisson(jr.PRNGKey(0), psf)
bg_noise = true_bg + jr.normal(jr.PRNGKey(0), psf_photon.shape)
image = psf_photon + bg_noise
data = image.flatten()

plt.figure(figsize=(15, 4))
plt.subplot(1, 2, 1)
plt.title("PSF")
plt.imshow(psf ** 0.25)
plt.colorbar()

plt.subplot(1, 2, 2),
plt.title("PSF + Photon")
plt.imshow(psf_photon ** 0.25)
plt.colorbar()
