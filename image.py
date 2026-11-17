import numpy as np
from scipy import ndimage


class Image:
    SHAPE = (104, 104, 104)
    CROP = 36

    def __init__(self, fwhm=3.0, rng=None):
        self.sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        self.rng = rng if rng is not None else np.random.default_rng()
        self.grid = np.zeros(self.SHAPE)

        self.scale = 1
        if self.sigma > 0:
            spike = np.zeros((41, 41, 41))
            spike[20, 20, 20] = 1
            kernel = ndimage.gaussian_filter(spike, self.sigma)
            self.scale = np.sqrt(np.sum(kernel ** 2))

    def add_noise(self):
        self.grid = self.rng.standard_normal(self.SHAPE)

    def add_signal(self, centre, radius, amplitude):
        c = self.CROP
        centre = (centre[0] + c, centre[1] + c, centre[2] + c)
        self.grid = self.grid + amplitude * self.scale * sphere_mask(self.SHAPE, centre, radius)

    def smooth(self):
        if self.sigma > 0:
            self.grid = ndimage.gaussian_filter(self.grid, self.sigma)

    def crop(self):
        c = self.CROP
        self.grid = self.grid[c:-c, c:-c, c:-c]

    def rescale(self):
        self.grid = self.grid / self.scale

    def generate(self, signal=None):
        self.add_noise()
        if signal is not None:
            self.add_signal(signal["centre"], signal["radius"], signal["amplitude"])
        self.smooth()
        self.crop()
        self.rescale()
        return self.grid


def sphere_mask(shape, centre, radius):
    idx = np.indices(shape)
    dist2 = sum((idx[d] - centre[d]) ** 2 for d in range(3))
    return dist2 <= radius ** 2
