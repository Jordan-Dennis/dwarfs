import equinox
import dLux
import typing
import jax.numpy as np
import matplotlib.pyplot as pyplot


Wavefront = typing.TypeVar("Wavefront")
Telescope = typing.TypeVar("Telescope")
Matrix = typing.TypeVar("Matrix")


class HubbleSpaceTelescope(equinox.Module):
    radius: float 
    mask_shift: float
    secondary_to_focus: float 
    primary_focal_length: float 
    primary_to_secondary: float 
    secondary_focal_length: float 
   

    def __init__(
            self: Telescope, /,
            radius: float = 1.2, 
            mask_shift: float = -.08,
            secondary_to_focus: float  = 6.3919974, 
            primary_focal_length: float = 5.52085,
            primary_to_secondary: float = 4.907028205,
            secondary_focal_length: float = -.6790325) -> Telescope:
        """
        Parameters
        ----------
        radius: float, meters
            The radius of the primary pupil/aperture. 
        mask_shift: float
            The shift of the mask in percentage of width.
        secondary_to_focus: float, meters
            The distace from the secindary to the focus. 
        primary_focal_length: float, meters
            The focal length of the primary mirror on the telescope.
        primary_to_secondary: float, meters
            The distance between the primary and secondary mirror.
        secondary_focal_length: float, meters
            The focal length of the secondary telescope mirror.
        """
        self.radius = radius
        self.mask_shift = mask_shift
        self.secondary_to_focus = secondary_to_focus
        self.primary_focal_length = primary_focal_length
        self.primary_to_secondary = primary_to_secondary
        self.secondary_focal_length = secondary_focal_length


    def pupil(self, npix: int) -> Matrix:
        """
        Construct a model of the pupil.

        Parameters
        ----------
        npix: int
            The number of pixels to use in representing the pupil.

        Returns
        -------
        pupil: float
            The pupil represented as an array. 
        """
        pupil_radius = 1.2
        obstructions_radius = 0.396
        spider_width = 0.20
        
        pixel_scale = 2 * pupil_radius / npix    
        cartesian = get_pixel_positions(npix, 0., 0.) * pixel_scale
        radial = np.hypot(cartesian[1], cartesian[0])

        horizontal_spider = (np.abs(cartesian[0]) <= spider_width)
        vertical_spider = (np.abs(cartesian[1]) <= spider_width)
        outer_edge = (radial <= pupil_radius)
        obstruction = (radial >= obstruction_radius)

        pupil = np.zeros_like(radial)\
            .at[outer_edge].set(1.)\
            .at[horizontal_spider].set(0.)\
            .at[vertical_spider].set(0.)\
            .at[obstruction].set(0.)

        return pupil
        
         
    def nicmos(self, npix: int) -> Matrix:
        """
        Constructs a detailed model of the nicmos1 camera pupil on the hubble 
        space telescope. 

        Parameters
        ----------
        npix: int
            The number of pixels along one edge of the image that represents the 
            camera pupil.

        Returns
        -------
        nicmos: Matrix
            The pupil of the nicmos1 camera represented as a binary array.
        """
        pix_scale = 2 / npix 
        mask_shift = int(self.mask_shift * npix / 2)
        cartesian = dLux.utils.get_pixel_positions(npix + mask_shift, 0., 0.) * pix_scale
        radial = (dLux.utils.get_polar_positions(npix + mask_shift, 0., 0.) * pix_scale)[0]
       
        pad_radius = .065
        pad_1_centre = int(.89221 * npix / 2) # In units of pixels
        pad_2_centre = int(.7555 * npix / 2), int(-.4615 * npix / 2)
        pad_3_centre = int(-.7606 * npix / 2), int(-.4564 * npix / 2)

        # TODO: Do we want the spider width ect. to be learnable. 
        # I don't think so
        spider_width = .011
        obstruction_width = .33
        outer_radius = (radial <= 1)
        telescope_obstruction = (radial <= obstruction_width)
        horizontal_spider = (np.abs(cartesian[0]) < spider_width)
        vertical_spider = (np.abs(cartesian[1]) < spider_width)
        mirror_pad_1 = np.roll(radial, pad_1_centre, axis=1) <= pad_radius
        mirror_pad_2 = np.roll(radial, pad_2_centre, axis=(0, 1)) <= pad_radius
        mirror_pad_3 = np.roll(radial, pad_3_centre, axis=(0, 1)) <= pad_radius

        optical_telescope_assembly = np.zeros_like(radial)\
            .at[outer_radius].set(1.)\
            .at[telescope_obstruction].set(0.)\
            .at[horizontal_spider].set(0.)\
            .at[vertical_spider].set(0.)\
            .at[mirror_pad_1].set(0.)\
            .at[mirror_pad_2].set(0.)\
            .at[mirror_pad_3].set(0.)

        nicmos_obstruction_width = .372
        nicmos_spider_width = .0335
        outer_radius = (radial <= .955)
        obstruction = (radial <= nicmos_obstruction_width)
        vertical_spider = (np.abs(cartesian[0]) < nicmos_spider_width)
        horizontal_spider = (np.abs(cartesian[1]) < nicmos_spider_width)

        inner = .065 
        outer = .8271
        unrotated_x = cartesian[0]
        unrotated_y = np.abs(cartesian[1])
        rotated_right_x = rotate(cartesian[0], 121. * np.pi / 180)
        rotated_right_y = np.abs(rotate(cartesian[1], 121. * np.pi / 180))
        rotated_left_x = rotate(cartesian[0], -121.5 * np.pi / 180)
        rotated_left_y = np.abs(rotate(cartesian[1], -121.5 * np.pi / 180))
        
        mirror_pad_1 = (unrotated_x >= outer) * (unrotated_y <= inner)
        mirror_pad_2 = (rotated_right_x >= outer) * (rotated_right_y <= inner)
        mirror_pad_3 = (rotated_left_x >= outer) * (rotated_left_y <= inner)
     
        nicmos_cold_mask = np.zeros_like(radial)\
            .at[outer_radius].set(1.)\
            .at[obstruction].set(0.)\
            .at[vertical_spider].set(0.)\
            .at[horizontal_spider].set(0.)\
            .at[mirror_pad_1].set(0.)\
            .at[mirror_pad_2].set(0.)\
            .at[mirror_pad_3].set(0.)

        nicmos_cold_mask = np.roll(nicmos_cold_mask, mask_shift, axis=0)

        return optical_telescope_assembly * nicmos_cold_mask
    

    def __call__(self: Telescope, wave: Wavefront):
        npix = wavefront.number_of_pixels()
        pupil = self.pupil(npix)
        nicmos = self.nicmos(npix)

        wave = wave.update_phasor(
            wave.get_amplitude() * pupil,
            wave.get_field() * pupil)

        wave = dLux.GaussianLens(primary_focal_length)(wave)
        wave = dLux.GaussianPropagator(primary_to_secondary)(wave)
        wave = dLux.GaussianLens(secondary_focal_length)(wave)
        wave = dLux.GaussianPropagator(secondary_to_focus)(wave)

        wave = wave.update_phasor(
            wave.get_amplitude() * nicmos,
            wave.get_phase() * nicmos)

        return wave


hubble = HubbleSpaceTelescope()

pyplot.imshow(hubble.nicmos(1024))
pyplot.colorbar()
pyplot.show()
    
