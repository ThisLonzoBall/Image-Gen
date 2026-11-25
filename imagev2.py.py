import numpy as np
from scipy import ndimage


def make_image(rng, image_size, fwhm, padding, signal_centre, signal_radius, signal_strength):

    # Convert FWHM into sigma
    sigma = fwhm / np.sqrt(8 * np.log(2))

    # enlarged image that will be cropped aster smoothing (104 voxels according to hayasaka+ nichols)
    padded_size = image_size + padding * 2

    # stores the factor needed to rescale the sd within out image
    noise_scale = 1

    if sigma > 0:
        k = int(4 * sigma) # radius of gaussian filter for each voxel
        point_image = np.zeros((2*k+1, 2*k+1, 2*k+1)) #3d lattice of size k in each diescrtions
        point_image[k, k, k] = 1 # place single voxel at image centre

        blurred_point = ndimage.gaussian_filter(point_image, sigma) # smooth image to see effect of gaussian filter on a single value
        noise_scale = np.sqrt(np.sum(blurred_point ** 2)) # factor that gives reduction in noise sd

    # create 3D image which a standard normal (0,1) at each voxel
    image = rng.standard_normal((padded_size, padded_size, padded_size))

    if signal_strength != 0:
        # shift signal centre to centre of padded image
        padded_centre = (signal_centre[0] + padding,signal_centre[1] + padding, signal_centre[2] + padding)

        # dimensions of every within the the image
        x, y, z = np.indices(image.shape)

        # Calc distrance from the centre of the image for each voxel
        distance_from_centre = ((x-padded_centre[0])**2 +(y-padded_centre[1])**2 +(z-padded_centre[2])**2)

        signal_mask = distance_from_centre <= signal_radius**2 # voxel inside sphere
        image += signal_strength * noise_scale * signal_mask # adding singal sphere reweighted for change in sd

    # Gaussian smoothing
    if sigma > 0:
        image = ndimage.gaussian_filter(image, sigma)

    # Crop image
    image = image[padding:-padding, padding:-padding, padding:-padding]

    # Rescale Sd to 1 again
    return image / noise_scale

# Hayasaka and nichols use images of size 32 cropped from size 104 voxels padded image

# Generate images
def simulate(N=20, seed=0, size=32, fwhm=3.0, pad=36,
             radius = 5, centre=None, amplitude=0.0):

    rng = np.random.default_rng(seed)

    # centre of cropped image
    if centre is None:
        centre = (size//2, size//2, size//2)

    # Store images
    images = []

    # create N images
    for i in range(N):
        image = make_image(rng, size, fwhm, pad, centre, radius, amplitude)
        images.append(image)

    #convert image to array
    image_data = np.array(images)

    # sd across images
    standard_deviation = image_data.std(axis=0, ddof=1)

    if amplitude != 0:
        x, y, z = np.indices((size, size, size))

        #signal centre
        distance_from_centre = ((x-centre[0])**2 +(y-centre[1])**2 +(z-centre[2])**2)

        # identify voxels within signal spehere
        true_signal = distance_from_centre <= radius**2

    else:
        # no signal
        true_signal = np.zeros((size, size, size), dtype=bool)

    return image_data, standard_deviation, true_signal