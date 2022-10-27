import matplotlib.pyplot as plt
import matplotlib as mpl
import hdfdict
import jax.numpy as np
import jax.random as jr
import dLux as dl
import chainconsumer as cc

from pupils import NicmosColdMask, HubblePupil

mpl.rcParams["text.usetex"] = True

chains = hdfdict.load("chains.hdf5")
chains = {key: np.array(chains[key]) for key in chains}
chains["focus"] = chains["coeffs"][:, 0]
chains["horizontal_astigmatism"] = chains["coeffs"][:, 1]
chains["diagonal_astigmatism"] = chains["coeffs"][:, 2]
chains["x_offset"] = chains["position"][:, 0]
chains["y_offset"] = chains["position"][:, 1]
chains.pop("position")
chains.pop("coeffs")
chains.pop("position_pix")
chains.pop("offset_x_latent")
chains.pop("offset_y_latent")
chains.pop("log_flux")
chains.pop("log_sep")
chains.pop("log_flux_ratio")

chain = cc.ChainConsumer()
chain.add_chain(chains)
chain.configure(serif=True, shade=True, bar_shade=True, shade_alpha=0.2, spacing=1., max_ticks=3)
fig = chain.plotter.plot()

coords = dl.utils.get_pixel_coordinates(128, 2.4 / 128)

true_position = np.zeros(2)
true_separation, true_field_angle = dl.utils.arcsec2rad(5e-1), 0
true_flux, true_flux_ratio = 1e5, 10
true_coeffs = jr.normal(jr.PRNGKey(0), (3,))
true_x_offset, true_y_offset = 0.067, -0.067
true_pixel_scale = dl.utils.arcsec2rad(0.043)
true_bg = 10.

aperture = dl.CompoundAperture(
    {"mask": NicmosColdMask(0., 0.), "pupil": HubblePupil()})
offset_aperture = dl.CompoundAperture(
    {"mask": NicmosColdMask(true_x_offset, true_y_offset), "pupil": HubblePupil()})

plt.imshow(aperture._aperture(coords))

plt.imshow(offset_aperture._aperture(coords))

with open("data/filters/HST_NICMOS1.F170M.dat") as filter_data:
    next(filter_data)
    nicmos_filter = np.array([
            [float(entry) for entry in line.strip().split(" ")] 
                for line in filter_data])\
        .reshape(10, 80, 2)\
        .mean(axis=1)\
        .at[:, 0]\
        .mul(1e-9)\
        .at[:, 1]\
        .divide(100)

dl_nicmos_filter = dl.Filter(nicmos_filter[:, 0], nicmos_filter[:, 1])

plt.subplot(1, 2, 2)
plt.title("Input filter")
plt.scatter(dl_nicmos_filter.wavelengths, dl_nicmos_filter.throughput)
plt.show()

wavelengths = np.tile(1e-6*np.linspace(1.6, 1.8, 3), (2, 1))
weights = np.ones(wavelengths.shape)
combined_spectrum = dl.CombinedSpectrum(wavelengths, weights).normalise()

true_position = np.zeros(2)
true_separation, true_field_angle = dl.utils.arcsec2rad(5e-1), 0
true_flux, true_flux_ratio = 1e5, 10
resolved = [False, False]
binary_source = dl.BinarySource(true_position, true_flux, true_separation, 
                             true_field_angle, true_flux_ratio, 
                             combined_spectrum, resolved, name="Binary")

# +
# Construct Optical system
wf_npix = 128
det_npix = 64

# Zernike aberrations,
basis = dl.utils.zernike_basis(6, npix=wf_npix)[3:] * 1e-9
true_coeffs = jr.normal(jr.PRNGKey(0), (basis.shape[0],))

pupils = {"Hubble": HubblePupil(), "Nicmos": NicmosColdMask(true_x_offset, true_y_offset)}

# Construct optical layers,
true_pixel_scale = dl.utils.arcsec2rad(0.043)
layers = [dl.CreateWavefront(wf_npix, 2.4, wavefront_type="Angular"),
          dl.TiltWavefront(),
          dl.CompoundAperture(pupils),
          dl.ApplyBasisOPD(basis, true_coeffs),
          dl.NormaliseWavefront(),
          dl.AngularMFT(true_pixel_scale, det_npix)]

# +
# Construct Detector,
true_bg = 10.
true_pixel_response = 1 + 0.05*jr.normal(jr.PRNGKey(0), (det_npix, det_npix))
detector_layers = [
    dl.AddConstant(true_bg),
    # dl.ApplyPixelResponse(true_pixel_response),
]

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
