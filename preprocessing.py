import numpy as np
from PIL import Image, ImageOps

def map_mask(gray_mask):
    """ Map the LabelMe label to the same encoding 

    Args:
        gray_mask (Image as Numpy Array): The grey-scale mask image

    Returns:
        Numpy Array: The mapped mask
    """    

    ## The grey-values of the different segments 
    ## Produced from LabelME
    label_bg = 0 
    label_summer = 14
    label_winter = 38    
    label_focus = 75
    label_river = 113

    #The mask codes considered by the generator
    mask_focus = 0  
    mask_summer = 1   
    mask_river = 2 
    mask_winter = 3 
    mask_bg = 4      
  
    #Save both encoding in lists to facilitate itteration
    label_codes = [label_bg, label_summer, label_winter, label_river, label_focus]
    mask_codes = [mask_bg, mask_summer, mask_winter, mask_river, mask_focus]

    
    mapped_mask = gray_mask.copy()
    ## Replace the label codes with the generated mask codes
    for i in range(len(label_codes)):
        mapped_mask [ gray_mask == label_codes[i]] = mask_codes[i]    
    
    
    return mapped_mask
    


def read_image(img_path, width=512, height=512):
    """Read an image from the disk and resize it to 512x512

    Args:
        img_path (String): The image path

    Returns:
        Numpy Array: The image
    """    
    img = Image.open(img_path)
    img = img.resize((width,height), 0)
    img = np.asarray(ImageOps.grayscale(img))

    return img

def read_and_map_mask(path):
    """Read an annotated map and map it to the same encoding of the gererated mask

    Args:
        path (string): the file path

    Returns:
        [Numpy Array]: The mapped mask
    """    
    gray_mask = read_image(path)
    mapped_mask = map_mask(gray_mask)
    
    return mapped_mask

def scale_byte_imgs(img):
    """Scale the grey-scale array into float between 0-1

    Args:
        img (Numpy Array]): The image

    Returns:
        image: The scaled image
    """    
    scaled_img = img / 255.0
    
    return scaled_img


def generate_eight(image):
    """Generate 8 images from one image by transpose and three rotations each

    Args:
        image (Numpy Array): an image to augment

    Returns:
        List: Augemnted eight images
    """    
    images = []
    #Append the image and its transpose to the list
    images.append(image)
    images.append(image.T)

    #Rotate the images and its transpose 3 times each time 90 degree
    #And add the rotated images to the list
    for i in range(3):
        image = np.rot90(image)
        images.append(image)
        images.append(image.T)
        
    return images


def augment_eight_scales_and_masks(scale, mask):
    """Augment a scale and its mask into eight images

    Args:
        scale (Image): The scale to augment
        mask (Image): The mask to augment

    Returns:
        Array: return eight scale images and 8 mask images
    """    
    batch_size = 8
    ## Create array of 8 images
    scale_batch = np.zeros((batch_size, scale.shape[0], scale.shape[1]), dtype=np.float32)
    mask_batch = np.zeros((batch_size, scale.shape[0], scale.shape[1]))
    
    # Augment the scale and the mask
    scales = generate_eight(scale)
    masks = generate_eight(mask)

    #Save the augmented images into array to include in the training dataset
    for i in range(8):
        scale_batch[i,:,:] = scales[i]
        mask_batch[i,:,:] = masks[i]
        
    return scale_batch, mask_batch


def read_data_set(data_path, IMG_COUNT, IMG_HEIGHT, IMG_WIDTH, label_suffix):
    """read the whole dataset

    Args:
        data_path (string): the main folder containing the data
        IMG_COUNT (int): Number of images to read
        IMG_HEIGHT (int): the image height
        IMG_WIDTH (int): images width
        label_suffix (string): mask or label (the first for generated and second of labelled data)

    Returns:
        two arrays: The scales and their labels
    """    
    #Create arrays to save the datasets
    scales = np.zeros((IMG_COUNT, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    masks = np.zeros((IMG_COUNT, IMG_HEIGHT, IMG_WIDTH), dtype = np.ubyte)

    #Loop through images and read them with their labels
    for i in range(0, IMG_COUNT):

        scale = read_image(data_path + '/scale_' + str(i+1) + '.png')
        mask =  read_image(data_path + '/scale_' + str(i+1) + '_'+ label_suffix + '.png')
        scales[i] = scale
        if label_suffix == "label":
            mask = np.rint(map_mask(mask))
        ## Convert the mask coding to integer in the range between 0-4
        else:
            mask = mask / 64
        masks[i] = mask
    
    ## Scale the dataset 
    scales =   scale_byte_imgs(scales)
    
    return scales, masks

   

def augment_data_set(scales, masks):
    """Augment the dataset by creating 8 copies of each image transposed and rotated

    Args:
        scales (Array of images): The scales
        masks (Array of images): [The masks

    Returns:
        Two arrays: the augmeted scales and the augmented masks
    """    
    ##the data is augmented 8 times  
    augment_times = 8
    
    #Create two arrays for the augmented dataset and its labels
    augmented_scales = np.zeros((scales.shape[0]*augment_times, 
                         scales.shape[1], 
                         scales.shape[2]), 
                        dtype = np.float32)
    augmented_masks = np.zeros((scales.shape[0]*augment_times, 
                         scales.shape[1], 
                         scales.shape[2]), 
                        dtype = np.ubyte)

    #Augment each scale and its scale then save the augmented data
    for i in range(scales.shape[0]):
        eight_scales, eight_masks = augment_eight_scales_and_masks(scales[i], masks[i])
        augmented_scales[i * augment_times:(i+1) * augment_times,:,:]  = eight_scales
        augmented_masks[i * augment_times:(i+1) * augment_times,:,:]  = eight_masks
        
    return augmented_scales, augmented_masks
