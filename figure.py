import matplotlib.pyplot as plt
import matplotlib as mpl
import hdfdict
import jax.numpy as np
import jax.random as jr
import dLux as dl
import chainconsumer as cc

from pupils import NicmosColdMask, HubblePupil

mpl.rcParams["text.usetex"] = True
mpl.rcParams['image.cmap'] = 'inferno'
mpl.rcParams["font.family"] = 'serif'
mpl.rcParams["text.usetex"] = 'true'
mpl.rcParams['figure.dpi'] = 120

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

wf_npix = 128
det_npix = 64

basis = dl.utils.zernike_basis(6, npix=wf_npix)[3:] * 1e-9
true_coeffs = jr.normal(jr.PRNGKey(0), (basis.shape[0],))

pupils = {"Hubble": HubblePupil(), "Nicmos": NicmosColdMask(true_x_offset, true_y_offset)}
naive_pupils = {"Hubble": HubblePupil(), "Nicmos": NicmosColdMask(0., 0.)}

true_pixel_scale = dl.utils.arcsec2rad(0.043)

layers = [dl.CreateWavefront(wf_npix, 2.4, wavefront_type="Angular"),
    dl.TiltWavefront(),
    dl.CompoundAperture(pupils),
    dl.ApplyBasisOPD(basis, true_coeffs),
    dl.NormaliseWavefront(),
    dl.AngularMFT(true_pixel_scale, det_npix)]

naive_layers = [dl.CreateWavefront(wf_npix, 2.4, wavefront_type="Angular"),
    dl.TiltWavefront(),
    dl.CompoundAperture(pupils),
    dl.NormaliseWavefront(),
    dl.AngularMFT(true_pixel_scale, det_npix)]

telescope = dl.Telescope(
     dl.Optics(layers), 
     dl.Scene([binary_source]),
     filter=dl_nicmos_filter,
     detector=dl.Detector([dl.AddConstant(true_bg)]))

naive_telescope = dl.Telescope(
    dl.Optics(naive_layers), 
    dl.Scene([binary_source]),
    filter=dl_nicmos_filter)

## Gerenate psf,
psf = telescope.model_scene()
naive_psf = naive_telescope.model_scene()
bg_noise = true_bg #+ jr.normal(jr.PRNGKey(0), psf_photon.shape)
data = psf + bg_noise

import matplotlib.cm as cm

mappables = []

# So I need to add the plots of the aperture but I don't know if it will be in the same figure. 

# +
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes[0][0].set_title("PSF")
mappables.append(axes[0][0].imshow(psf ** 0.25))
axes[0][0].set_xticks([])
axes[0][0].set_yticks([])
plt.colorbar(mappables[-1], fraction=0.045, ax=axes[0][0])

axes[0][1].set_title("Residuals")
mappables.append(axes[0][1].imshow(abs(data - psf) ** 0.25))
axes[0][1].set_xticks([])
axes[0][1].set_yticks([])
plt.colorbar(mappables[-1], fraction=0.045, ax=axes[0][1])

axes[0][2].set_title("Data")
mappables.append(axes[0][2].imshow(data ** 0.25))
axes[0][2].set_xticks([])
axes[0][2].set_yticks([])
plt.colorbar(mappables[-1], fraction=0.045, ax=axes[0][2])


plt.subplot(3, 3, 6)
plt.title("Niave Model")
plt.imshow(naive_psf ** 0.25)
plt.xticks([])
plt.yticks([])
plt.colorbar()


plt.subplot(3, 3, 9)
plt.title("Residuals")
plt.imshow(abs(data - naive_psf) ** 0.25)
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.tight_layout()
plt.show()
# -

# Now I have all the relevant tools to create the figure at my disposal however, I am going to head home now so that I can get everything that I need to get done in my life done. There will probably be some more work on this after dinner. 


