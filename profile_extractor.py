import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.draw import line

river_code = 126
path = 'HDA/tr/scale_'

plt.rcParams['figure.figsize'] = 30,20
plt.rcParams.update({'font.size': 22})

data_path = 'profile_extractor_data/'
result_path = 'profile_extractor_result/'




def get_four_corners(pixels):
    
    """Gets the scale pixels that are not background a
       Finds the four corner pixels

    Parameters
    ----------
    pixels : list of pixels
        all the scale pixels
    
    Returns
    -------
    left_pixel
        the most left pixel
    right_pixel
        the mose right pixel
    top_pixel
    
    bottom_pixel
    """
    
    #Save teh indices into data frame to sort them
    pixels_data_frame = pd.DataFrame()
    
    pixels_data_frame['Column'] = pixels[1]
    pixels_data_frame['Row'] = pixels[0]
    #Sort the data frame by row to get the top and bottom pixels
  
    pixels_data_frame = pixels_data_frame.sort_values('Row')
    top_pixel = pixels_data_frame.iloc[0] .values  
    bottom_pixel = pixels_data_frame.iloc[-1].values
   
    #Sort the dataframe by column to get the left and right pixels
    
    pixels_data_frame = pixels_data_frame.sort_values('Column')
    left_pixel = pixels_data_frame.iloc[0].values
    right_pixel = pixels_data_frame.iloc[-1].values
   
    
    return left_pixel, right_pixel, top_pixel, bottom_pixel




def measure_distance(point1, point2):
        
    """Gets tow pointgs
       Returns the geometric distance between p1 and p2 
          
    Parameters
    ----------
    point1 : 2 elements array
        the first point
    point1 : 2 elements array
        the second point
    Returns
    -------
    distance
        the distance between the two points
    """
    ## distance = sqrt ( (x2-x1)^2 + (y2-y1)^2 )
    distance = ((((point1[0] - point2[0])**2) + ((point1[1] - point2[1])**2))**0.5)
    
    return distance


def select_furthest_point(centre, left, right, top, bottom):
    """Gets the centre and the edge four points
       Returns the geometric distance between p1 and p2 
          
    Parameters
    ----------
    centre: point
        the center of the scale
    left, right, top, bottom: 
        4 points
    Returns
    -------
    anterior
        the furthest point out of the four is the anterior
    """    
    #Save the points in a list to itterate through them
    corner_points = [left, right, top, bottom]
    points_distances = []
    
    #Measure the distance between the centre and every corner
    for i in range(len(corner_points)):
        points_distances.append(measure_distance(centre, corner_points[i]))
    
    # the anterior is the point that has the largest distance from the centre
    anterior = corner_points[points_distances.index(max(points_distances))]
    return anterior

 
def drow_circles(img, centre, points, radius, color, line_width):
    """Drow the circules of the center and the four corners

    Args:
        img (img): The image
        centre (point): the river growth center
        points (list of points): the four furthest points of the centre
        radius (int): the circle radius
        color (byte): circle lines color 0-255
        line_width (byte): Line width

    Returns:
        img: The image with all circles plotted on it
    """    
    
    points.append(centre)
    for point in points:
        cv.circle(img, point, radius, color, line_width)
        
    return img

def drow_lines(img, centre, lines, color, line_width):
    """Drow lines between the center and every corner

    Args:
         img (image): The image
        centre (point): the river growth center
        lines (list of points): the four furthest points of the centre
        color (byte): circle lines color 0-255
        line_width (byte): Line width

    Returns:
        img: The image with lines plotted on it
    """    
    for line in lines:
        cv.line(img, centre, line, color, line_width)
        
    return img
    

def drow_rays(mask, scale, only_anterior=False):
    """Show the center and the four corners of the scale

    Args:
        mask (img): The predicted mask
        scale (img): The original scale
        only_anterior (bool, optional): Plot only the longest line or all. Defaults to False.

    Returns:
        [type]: [description]
    """    
    
    gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    #Extract river pixels
    river_pixels = np.where(gray == river_code)
    scale_pixels = np.where(gray <= 250 )
    
    gray_scale = cv.cvtColor(scale, cv.COLOR_BGR2GRAY)

    #Get the four corners
    left, right, top, bottom = get_four_corners(scale_pixels)
    # Decide the center
    centre_row = (int(np.median(river_pixels[0]))) + int(np.median(river_pixels[0]) / 30)
    centre_column = (int(np.median(river_pixels[1])))
    
    centre = (centre_column,centre_row)
    
    # Get the anterior from the furthest points
    anterior =  select_furthest_point(centre, left, right, top, bottom)  
    #Draw the circles
    radius, color, width = 20, 100, 2
    line_width, line_color = 2, 10
    
    if not only_anterior:        
        circles = [centre, bottom, top, left, right]
    else:
        circles = [anterior]
    
    
    gray_scale = drow_circles(gray_scale, centre, circles, radius, color, width) 

    #Draw the lines   
    gray_scale = drow_lines(gray_scale, centre, circles, line_color, line_width)
 
    
    return gray_scale

def extract_circulies_profile(mask, scale):
    """Extract the ciruli prfoile and their corresponding predicted bands from the scale and mask

    Args:
        mask (img): The predicted mask
        scale (img): The original scale

    Returns:
        circulie_profile, mask_profile: The profile of circulies and their corresponding bands
    """    
    
    
    gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    # Filter the river pixel to find the centre
    river_pixels = np.where(gray == river_code)
    scale_pixels = np.where(gray < 250 )

    gray = cv.cvtColor(scale, cv.COLOR_BGR2GRAY)
    # Get the four corners of the scale
    left, right, top, bottom = get_four_corners(scale_pixels)
    # Find the centre
    centre_row = (int(np.median(river_pixels[0]))) + int(np.median(river_pixels[0]) / 25)
    centre_column = (int(np.median(river_pixels[1])))
    
    centre = (centre_column,centre_row)

    #Get the anterior
    anterior =  select_furthest_point(centre, left, right, top, bottom)  
    #Get the anterior pixel indices 
    cc, rr = line(centre_column, centre_row, anterior[0], anterior[1])
    # Extract the scale profile and mask profile
    mask_profile = mask[rr,cc]
    scale_profile = scale[rr,cc]
    
    return mask_profile, scale_profile


def read_image(path):
    img = cv.imread(path)
   
    return img
   
    
def save_img(path, img):
    cv.imwrite(path, img)
    
def show_img(img):
    plt.figure()
    plt.imshow(img, cmap='gray')
    
def get_profile_areas_color(mask_ray):
    """Trace the mask to determine where each band starts and end

    Args:
        mask_ray (array): The anterior ray of mask pixels

    Returns:
        2D array: decides where every band starts and the band colour
    """    
    areas = []

    #Record the first colour
    current_color = mask_ray[0]
    areas.append([0,current_color])

    # Iterate through all the pixel
    for i in range(1, len(mask_ray), 1):
        #If the colour change, record the new colour and its starting index
        if mask_ray[i] != current_color:
            current_color = mask_ray[i]
            areas.append([i,current_color])
        #Record the end of the rays
        if i == len(mask_ray) - 1 :
            areas.append([i,current_color])
            
    return areas    

def plot_profile(scale_profile, bands_edges, save_path =""):
    """Plot the circuli profile and colour background 
    based on the start and the end of each band

    Args:
        scale_profile (Scale ray): Scale anterior ray of pixels
        bands_edges (array): shows where each band start and its color
        save_path (str, optional): path to save the file. Defaults to "".
    """    
    plt.figure()    
    
    for j in range(0,len(bands_edges) - 1):
        start = int(bands_edges[j][0])
        end = int(areas[j+1][0])
        #print(start,bands_edges)
        color = int(bands_edges[j][1])/4
        #print(a[0])
        plt.axvspan(start,end,  facecolor=(color,color,color), alpha = 0.99, zorder=-100)
        
    plt.plot(scale_profile, lw=1, c='b')

    #Save the figure if there is a path
    
    if save_path != "":
        plt.savefig(save_path)


for i in range(4):
    scale = read_image(data_path + 'scale_' + str(i) + ".png")
    mask = read_image(data_path + 'scale_' +str(i) + "_mask_pred.png")
    scale_with_rays = drow_rays(mask,scale)
    mask_with_rays = drow_rays(mask,mask)
    save_img(result_path + 'scale' + str(i) + "_profile_rays.png", scale_with_rays)
    save_img(result_path + 'scale' + str(i) + "_profile_rays_mask.png", mask_with_rays)
    maskp , scalep = extract_circulies_profile(mask, scale)   
    areas = get_profile_areas_color(maskp[:,0]/63)
    plot_profile(scalep, areas, result_path +'scale' + str(i) + "_profile.png")
