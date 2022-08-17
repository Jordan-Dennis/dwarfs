def affine(image: matrix, transformation: matrix, offset: float=0.,
        output_shape: tuple) -> matrix:
    output = np.zeros(output_shape, dtype=image.dtype)
    
    


def rotate(image: matrix, angle: float, axes: tuple = (0, 1)) -> matrix:
    axes = axes\
        .at[np.asarray(axes < 0).astype(int)]\
        .mul(image.ndim)\
        .sort()

    rotation_matrix = np.array([
        [np.cos(angle), np.sin(angle)], 
        [-np.sin(angle), np.cos(angle)]])

    image_shape = np.asarray(image.shape)
    input_plane_shape = image_shape[axes]

    centre = (input_plane_shape - 1) / 2
    
