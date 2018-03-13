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
    #Situation with no overlap on final tile;
    if current_step+step_size+padding == length:
        return steps, [step+padding-remainder for step in steps]
    else:
        final_step = length - step_size - padding - remainder
        steps.append(final_step)
        return steps, [step+padding for step in steps]
    
def crop_array(input_array, xlength, ylength=None, orgn=(0,0)):
    """Crops an image in numpy array format. Pads crops outside
    of input image with zeros if necessary. If no y dimension
    is specified, outputs a square image.
    """
    img = Image.fromarray(input_array)
    if ylength == None:
        ylength = xlength
    xmin = orgn[1]
    ymin = orgn[0]
    xmax = xmin + xlength
    ymax = ymin + ylength
    cropped = img.crop((xmin, ymin, xmax, ymax))
    return np.array(cropped)

def crop_crater(image, proposal, dim=(12, 12), px=4):
    """Takes an input image of arbitrary size X, Y and a
    proposal, which is a tuple (x, y, d) of a crater where
    x < X and y < Y. Returns a cropped image of 
    the proposal. The crater needs to be centered in the image.
    """
    x = proposal[0] #Record x and y positions
    y = proposal[1]
    #"Radius" of image
    r_im = proposal[2]*min(dim)/(2*px)
    #get four parameters of image box
    left = x - r_im
    upper = y - r_im
    right = x + r_im
    lower = y + r_im
    img = Image.fromarray(image)
    cropped = img.crop((left, upper, right, lower))
    if cropped.size != dim:
        cropped = cropped.resize(dim)
    return np.array(cropped)

def resolve_color_channels(image, desired=1):
    """Converts an image to the desired number of color
    channels. Returns converted image.
    """
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
        bottom='off', 
        top='off', 
        labelbottom='off', 
        right='off', 
        left='off', 
        labelleft='off'
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