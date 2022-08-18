import jax.numpy as np
from scipy import fftpack


def _cartesian_to_polar(x: matrix, y: matrix)


def _rotate(image: matrix, rotation: float):
    npix = image.shape[0]
    centre = (npix - 1) / 2
    x_pixels, y_pixels = get_pixel_positions(npix)
    rs, phis = cart2polar(x_pixels, y_pixels)
    phis += rotation
    coordinates_rot = np.roll(polar2cart(rs, phis) + centre, shift=1, axis=0)
    rotated = map_coordinates(array, coordinates_rot, order=order)
    return rotated


def fft_3shear_rotate_pad(in_frame, alpha, pad=4, return_full=False):
	# We need to add some extra rows since np.rot90 has a different definition of the centre
    in_shape = in_frame.shape
    image_shape = map(lambda x: x + 3, in_shape)
	image = np.full(shape, np.nan, dtype=float)\
        .at[1 : in_shape[0] + 1, 1 : in_shape[1] + 1]\
        .set(in_frame)

	# FFT rotation only work in the -45:+45 range
    # So I need to work out how to determine the quadrant that alpha is in and hence the 
    # number of required pi/2 rotations and angle in radians. 
    half_pi_to_1st_quadrant = (alpha + np.pi / 4) // (np.pi / 2)
    angle_in_first_quadrant = angle - (half_pi_to_1st_quadrant * np.pi / 2)
    
    image = np.rot90(image, half_pi_to_1st_quadrant)\
        .at[:-1, :-1]\
        .get()	

    width, height = np.array(image.shape, dtype=int)
	# Calculate the position that the input array will be in the padded array to simplify
	#  some lines of code later 
    # NOTE: This is the location to which I have reached. 
    left_corner = int(((pad - 1) / 2.) * width)
	right_corner = int(((pad + 1) / 2.) * width)
	top_corner = int(((pad - 1) / 2.) * height)
	bottom_corner = int(((pad + 1) / 2.) * height)

	# Make the padded array	
    out_shape = (width * pad, height * pad)
	padded_image = np.full(out_shape, np.nan, dtype=float)\
        .at[left_corner : right_corner, top_corner : bottom_corner]\
        .set(image)

	pad_mask = np.ones(out_shape, dtype=bool)
	pad_mask[px1:px2,py1:py2]=np.where(np.isnan(in_frame),True,False)
	
	# Rotate the mask, to know what part is actually the image
	pad_mask=ndimage.interpolation.rotate(pad_mask, np.rad2deg(-alpha_rad),
		  reshape=False, order=0, mode='constant', cval=True, prefilter=False)

	# Replace part outside the image which are NaN by 0, and go into Fourier space.
	pad_frame=np.where(np.isnan(pad_frame),0.,pad_frame)


	###############################
	# Rotation in Fourier space
	###############################
	a=np.tan(alpha_rad/2.)
	b=-np.sin(alpha_rad)

	M=-2j*np.pi*np.ones(pad_frame.shape)
	N=fftpack.fftfreq(pad_frame.shape[0])

	X=np.arange(-pad_frame.shape[0]/2.,pad_frame.shape[0]/2.)#/pad_frame.shape[0]

	pad_x=fftpack.ifft((fftpack.fft(pad_frame, axis=0,overwrite_x=True).T*
		np.exp(a*((M*N).T*X).T)).T, axis=0,overwrite_x=True)
	pad_xy=fftpack.ifft(fftpack.fft(pad_x,axis=1,overwrite_x=True)*
		np.exp(b*(M*X).T*N), axis=1,overwrite_x=True)
	pad_xyx=fftpack.ifft((fftpack.fft(pad_xy, axis=0,overwrite_x=True).T*
		np.exp(a*((M*N).T*X).T)).T,axis=0,overwrite_x=True)

	# Go back to real space
	# Put back to NaN pixels outside the image.

	pad_xyx[pad_mask]=np.NaN


	if return_full:
		return np.real(pad_xyx).copy()
	else:
		return np.real(pad_xyx[px1:px2,py1:py2]).copy()
