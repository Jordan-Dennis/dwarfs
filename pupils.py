import dLux as dl
import equinox as eqx
import jax.numpy as np


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


