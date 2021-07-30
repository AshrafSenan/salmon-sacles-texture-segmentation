
from enum import Enum
import math
import numpy as np
import matplotlib
from skimage.draw import ellipse, ellipse_perimeter
import skimage
import pandas as pd
import matplotlib.pyplot as plt
import random
import threading

IMG_HEIGHT = 512
IMG_WIDTH = 512
data_path = 'drive/My Drive/Salmon_Scale/imgs'

## Set the main properties of the generator
spacing = {'river_average':8.54, 
           'river_stdv':3, 
           'summer_average':21, 
           'summer_stdv':5, 
           'winter_average':15, 
           'winter_stdv':3, 
           }  

circulies = { 'river_average':20, 
              'river_stdv':4,
              'summer_average':24, 
              'summer_stdv':6,
              'winter_average':10, 
              'winter_stdv':4   
           }

focus = {'length_average':27,
         'length_stdv':4,
         'width_average':19,
         'width_stdv':4}  


scale = {'length_average':1263,
         'length_stdv':147,
         'width_average':962,
         'width_stdv':124}


colors = {'circulie_average':52,
          'circulie_stdv':22,
          'bg_average': 165,
          'bg_stdv':13}




# Generate the random parameters for the model based on the settings

def create_random_parameters():
    """Create a random scale properties b
    ased on the spacing, circulies, focus, scale and colour

    Returns:
        [Dictionary]: The main parameters of the scale
    """    
    MIN_BANDS = 3
    MAX_BANDS = 5
    # Define focus length and width
    focus_length = int(random.uniform(focus['length_average'] - focus['length_stdv'], 
                                      focus['length_average'] + focus['length_stdv']))
    focus_width = int(random.uniform(focus['width_average'] - focus['width_stdv'], 
                                     focus['width_average'] + focus['width_stdv']))
    
    # Define the scale length and width
    scale_length = int(random.uniform(scale['length_average'] - scale['length_stdv'], 
                                      scale['length_average'] + scale['length_stdv']))
    
    scale_width = int(random.uniform(scale['width_average'] - scale['width_stdv'], 
                                     scale['width_average'] + scale['width_stdv']))
    
    ## Generate spacing
    river_spacing = round(random.uniform(spacing['river_average'] - spacing['river_stdv'], 
                                         spacing['river_average'] + spacing['river_stdv']),3)
    summer_spacing = round(random.uniform(spacing['summer_average'] - spacing['summer_stdv'], 
                                          spacing['summer_average'] + spacing['summer_stdv']),3)
    winter_spacing = round(random.uniform(spacing['winter_average'] - spacing['winter_stdv'], 
                                          spacing['winter_average'] + spacing['winter_stdv']),3)

    
    #Define number of bands
    number_of_bands = random.randint(MIN_BANDS ,MAX_BANDS)

    #Define the circuiles color and background color
    circulie_color = random.randint(colors['circulie_average'] - colors['circulie_stdv'],
                                    colors['circulie_average'] + colors['circulie_stdv'])
    
    bg_color = random.randint(colors['bg_average'] - colors['bg_stdv'],
                                    colors['bg_average'] + colors['bg_stdv'])
    
    ##Save the parameters to dictionary
    parameters_dictionary = {'focus_length': focus_length,
                             'focus_width': focus_width,
                             'scale_length': scale_length,
                             'scale_width': scale_width,
                             'river_spacing': river_spacing,
                             'summer_spacing': summer_spacing,
                             'winter_spacing': winter_spacing,
                             'number_of_bands': number_of_bands,
                             'circulie_color': circulie_color,
                             'bg_color': bg_color}

    return parameters_dictionary 



def create_random_scale():
    """Create the random bands and circulies 
    based on the generated main parameters 

    Returns:
        Dictionary: The scale main properties, the bands and bands circulies
    """    
    
    #Create random parameters based on statstics above
    random_parameters = create_random_parameters()
    
    #Get the main parameters
    number_of_bands = random_parameters["number_of_bands"]
    bands_names = ["river", "summer", "winter"]
    focus_length = random_parameters['focus_length']
    focus_width = random_parameters['focus_width']
    ##The codes for riber, summer, and winter

    bands_codes = [128,64,192]
    bands_circulies_data = []
    ##start at the edge of the focus
    last_circulie = focus_length

    #Create bands 
    for i in range(number_of_bands):
        circulies_data = []
        band_name = bands_names[i%3]
        band_code = 0

        #First band is river
        if i == 0:
            band_name = bands_names[0]
            band_code = bands_codes[0] 
        #Odd bands are summers
        elif i%2 == 1:
            band_name = bands_names[1]
            band_code = bands_codes[1]
        #Even bands are winter
        elif i%2 == 0:
            band_name = bands_names[2]
            band_code = bands_codes[2]


        # Get the bands spacing average and stdv    
        band_circulies_space_average = spacing[band_name+"_average"]
        band_circulies_space_stdv = spacing[band_name+"_stdv"]     
        band_circulies_count_average = circulies[band_name + "_average"]
        band_circulies_count_stdv = circulies[band_name + "_stdv"]
        # Decide the number of circulies
        number_of_band_circulies = random.randint(int(band_circulies_count_average - band_circulies_count_stdv), 
                                    int(band_circulies_count_average + band_circulies_count_stdv))
        

        #Create the bands circulie        
        for j in range(number_of_band_circulies):
            # determine the bands spacing and add some variation
            band_space = random.randint(int(band_circulies_space_average - band_circulies_space_stdv), 
                                    int(band_circulies_space_average + band_circulies_space_stdv))

            # The current circulus location = the previous circuls + spacing
            circulie_length = band_space + last_circulie

            # Append the circulus to the list
            circulies_data.append(circulie_length)
            # Update the last circuli to drow after it
            last_circulie = circulie_length
        ## Add the band into bands
        band_dic = {'band_code':band_code,
                   'band_circulies':circulies_data}
        
        bands_circulies_data.append(band_dic)
    
    #Create the final scale containg the main parameters and the bands        
    scale_dic = random_parameters
    scale_dic['bands'] = bands_circulies_data
    
    return scale_dic






def rasterizeScale(scale_data):
    """Rasterize the scale and its mask

    Args:
        scale_data (Dictionary): The scale main parameters, bands anc circuli

    Returns:
        image, and its mask: the scale image and mask used for training
    """     
    # Read the main parameters
    bands = scale_data['bands']
    scale_length = scale_data['scale_length']
    scale_width = scale_data['scale_width']
    circulie_color = scale_data['circulie_color']
    bg_color = scale_data['bg_color']
    width_to_length = scale_width / scale_length
    last_band = bands[-1]
    rotation = random.randint(-180,180)
    last_circulies = last_band['band_circulies'][-1]
    ## Create an images 3 times the size of the last circulie
    imge_size = (int(last_circulies * 3 ), int(last_circulies * 3 ))
    
    # Define the centre of the image
    centerX, centerY = imge_size[0]//2, imge_size[1]//2

    # Create an image with random background
    image = np.random.randint(low = bg_color - 15, high = bg_color + 15, size = imge_size, dtype=np.ubyte) 
    # Create a white image for the mask
    mask = np.ones(imge_size, np.ubyte) * 255
    
    # iterate throw bands and drow their circulies
    for band in bands:
        for circ in band['band_circulies']:
            #Add the circulie ellipse (2 pixels)
            rows, columns = ellipse_perimeter(centerX, centerY, 
                                                int(circ), 
                                                int(circ * width_to_length), 
                                                rotation, imge_size)
            image[rows, columns] = circulie_color
            rows1, columns1 = ellipse_perimeter(centerX, centerY, 
                                                int(circ)-1, 
                                                int(circ * width_to_length)-1, 
                                                rotation, imge_size)
            image[rows1, columns1] = circulie_color
            #Make the winter/summer circulies three pixel width
            if(band != bands[0]):                    
                
                rows1, columns1 = ellipse_perimeter(centerX, centerY, 
                                                int(circ)+1, 
                                                int(circ * width_to_length)-1, 
                                                rotation, imge_size)
                image[rows1, columns1] = circulie_color
    # Rasterize the mask using the last circuli of each band and the band code  
    for i in reversed(range(len(bands))):
        bandr = bands[i]
        last_circuli = bandr['band_circulies'][-1]
        rows2, columns2 = ellipse(centerX, centerY, last_circuli , last_circuli * width_to_length, imge_size, - rotation )
        mask[rows2, columns2] = bandr['band_code']
    # Plot the focus
    mask_code = 0
    bandf = bands[0]
    focus = bandf['band_circulies'][0]
    rows3, columns3 = ellipse(centerX, centerY, focus , focus * width_to_length, imge_size, -rotation )
    mask[rows3, columns3] = mask_code
    # Wave the image and its mask then 
    # Crop the top half of the image wit hadditional 80 pixel under the focus
    image, mask = wave(image, mask)
    image_halved = image[:centerX + 80, :]
    mask_halved = mask[:centerX + 80, :]
    
    # Decide where teh last circuli to crop the additional sides
    circ_pixels = np.where(image_halved < 80)
    top_row = min(circ_pixels[0])
    left_column = min(circ_pixels[1])
    right_column = max(circ_pixels[1])

    # Remove the extra space around the scale 
    image_final = image_halved[top_row: , left_column:right_column ]
    mask_final =  mask_halved[top_row:, left_column:right_column ]   
    return image_final, mask_final

def wave(img, mask, x_scaling = 8, y_scaling = 8, x_angle = 180, y_angle = 180):
        scale = img
        ## Select the scaling and rotating angle
        x_scaling = random.randint(3,8)
        y_scaling = random.randint(3,8)
        x_angle = random.randint(175,180)
        y_angle = random.randint(175,180)
        # The image shape
        rows, cols = img.shape[0], img.shape[1]

        ## Create two arrays for the waved images
        wavedScale = np.zeros(img.shape, dtype=np.ubyte) 
        wavedMask = np.zeros(img.shape, dtype=np.ubyte)

        ## Wave in x and y axes using the angles and strenth the waving by scaling
        for row in range(rows): 
            for col in range(cols):
                shift_x = 0
                shift_y = 0
                if x_angle != 0:
                    shift_x = int(x_scaling * math.sin(2 * np.pi * row / x_angle))
                if y_angle != 0:
                    shift_y = int(y_scaling * math.cos(2 * np.pi * col / y_angle)) 
                if row + shift_y < rows and col + shift_x < cols: 
                    wavedScale[row,col] = scale[(row+shift_y)%rows,(col+shift_x)%cols] 
                    wavedMask[row,col] = mask[(row+shift_y)%rows,(col+shift_x)%cols] 
                else: 
                    wavedScale[row,col] = random.randint(150, 180) #Background pixels
                    wavedMask[row,col] = 255 #Masks Background pixels

        return wavedScale, wavedMask




from PIL import Image, ImageOps

def save_image(img, angle, path):
    """Rotate the image by angle and save it to file

    Args:
        img (image): the image to save
        angle (int): The rotation angle
        path (string): file path
    """    
    im = Image.fromarray(np.uint8(img))
    im = im.resize((IMG_HEIGHT,IMG_WIDTH),0)    
    im = im.rotate(angle)
    im.save(path)
  



def generate_images(start, end):
    """[generate images from start to end i.e 1-5

    Args:
        start (int): starting index
        end (int): ending index
    """    
    for i in range(start, end, 1):
        # Create random scale
        rand = create_random_scale()
        path = 'generated_images'

        #Rasterize the image and it's scale
        img, mask = rasterizeScale(rand)
        scale_shape = img.shape
        print("Generating Scale:" + str(i))
        #Rotate the image
        if scale_shape[0] > scale_shape[1]:
            img = np.rot90(img)
            mask = np.rot90(mask)
        angles = [0, 90, 180, 270]
        angle = random.choice(angles)

        # Save the scale and mask images
        save_image(img, angle, path + "/Scale_" + str(i)  + ".png" )
        save_image(mask, angle, path + "/Scale_" + str(i)  + "_mask.png")
        

num_of_thread = 10
num_of_images = 10
images_per_thread = 1

def generate_n_images():
    """Generate 1000 random scales dataset using multiprocessing
    """    
    import multiprocessing
    
    for i in range(0, num_of_images, images_per_thread):
        p = multiprocessing.Process(target=generate_images, args=(i, i+images_per_thread))
        p.start()

#Call the generatring function       
generate_n_images()

