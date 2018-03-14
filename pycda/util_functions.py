import numpy as np
import PIL.Image as Image
from skimage import color
import time, sys

def get_steps(length, input_dimension, output_dimension):
    """Calculates each step along a dimension of the tile.
    length is the total length of the dimension, model resolution
    is the detector's expected input length, and padding is the model's
    required padding (zero if none.)
    """
    steps = []
    padding = (input_dimension - output_dimension)//2
    remainder = (input_dimension - output_dimension)%2
    step_size = output_dimension - remainder
    current_step = 0-padding
    steps.append(current_step)
    current_step += step_size
    #Iterate until final prediction "falls off" edge of input image
    while (steps[-1]+(2*step_size+padding)) < length:
        steps.append(current_step)
        current_step += step_size
    #Situation with no overlap on final tile or small length;
    if current_step+step_size+padding == length or length <= output_dimension:
        return steps, [step+padding-remainder for step in steps]
    else:
        final_step = length - step_size - padding - remainder
        steps.append(final_step)
        return steps, [step+padding for step in steps]
    
def crop_array(input_array, ylength, xlength=None, orgn=(0,0)):
    """Crops an image in numpy array format. Pads crops outside
    of input image with zeros if necessary. If no y dimension
    is specified, outputs a square image.
    """
    if xlength == None:
        xlength = ylength
    ylength = int(ylength)
    xlength = int(xlength)
    orgn = (int(orgn[0]), int(orgn[1]))
    target = np.zeros((ylength, xlength))
    #slice ranges
    ymin = max(orgn[0], 0)
    xmin = max(orgn[1], 0)
    ymax = min(orgn[0] + ylength, input_array.shape[0])
    xmax = min(orgn[1] + xlength, input_array.shape[1])
    yslice = slice(ymin, ymax)
    xslice = slice(xmin, xmax)
    #top, left, bottom, right pads
    tp = max(-orgn[0], 0)
    lp = max(-orgn[1], 0)
    bp = max((ylength + orgn[0] - tp - input_array.shape[0]), 0)
    rp = max((xlength + orgn[1] - lp - input_array.shape[1]), 0)
    #insert slice into the right spot.
    target[tp:(ylength-bp),lp:(xlength-rp)] = input_array[yslice, xslice]
    return target

def make_batch(image, crop_dims, crops, out_dims=None):
    """Assembles a batch for model."""
    if not isinstance(crop_dims, list):
        crop_dims = [crop_dims for x in range(len(crops))]
    batch = []
    for i, crop_coords in enumerate(crops):
        next_image = crop_array(image, crop_dims[i][0], crop_dims[i][1], crop_coords)
        if out_dims != None:
            if next_image.shape != out_dims:
                resized = Image.fromarray(next_image).resize((out_dims[1], out_dims[0]))
                next_image = np.array(resized)
        if len(next_image.shape) == 2:
            #add color channel to greyscale image
            next_image = np.expand_dims(next_image, axis=-1)
        if next_image.dtype == np.dtype('uint8'):
            #Rescale pixel values
            next_image = next_image/255
        batch.append(next_image)
    batch = np.array(batch)
    return batch

def get_crop_specs(proposal, classifier):
    """Converts a crater proposal into cropping function
    arguments.
    """
    lat = proposal[0]
    long = proposal[1]
    px = classifier.crater_pixels
    dim = classifier.input_dims
    #"Radius" of image
    r_im = proposal[2]*min(dim)/(2*px)
    #get four parameters of image box
    upper = lat - r_im
    left = long - r_im
    return (round(upper), round(left)), (round(2 * r_im), round(2 * r_im))

def resolve_color_channels(prediction, model):
    """Converts an image to the desired number of color
    channels. Returns converted image.
    """
    image = prediction.input_image.copy()
    desired = model.input_channels
    if len(image.shape) == 2:
        image_channels = 1
    else:
        image_channels = image.shape[2]
    if image_channels == desired:
        return image
    elif image_channels == 3 and desired == 1:
        return color.rgb2grey(image)
    elif image_channels == 1 and desired > 1:
        print('Working on feature to convert greyscale to RGB. '
              'Try using a greyscale detector.')
    raise Exception('The color channels of the input image are '
                    'not compatible with this model.'
                   'look for a model with the proper number of '
                   'color channels for your image.')
    return image

def remove_ticks(ax_obj):
    """takes an ax object from matplotlib and removes ticks."""
    ax_obj.tick_params(
        axis='both', 
        which='both', 
        bottom=False, 
        top=False, 
        labelbottom=False, 
        right=False, 
        left=False, 
        labelleft=False
        )
    return ax_obj


def update_progress(progress):
    """Displays or updates a console progress bar
    Accepts a float between 0 and 1. Any int will be converted to a float.
    A value under 0 represents a 'halt'.
    A value at 1 or bigger represents 100%
    """
    barLength = 25 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rProgress: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), round(progress*100), status)
    sys.stdout.write(text)
    sys.stdout.flush()