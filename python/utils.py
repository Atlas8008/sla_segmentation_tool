import numpy as np

def area_from_segmentation_map(segmentation_map, dpi):
    dpcm = dpi / 2.54 # Dots per cm    
    
    pixel_area = np.sum(segmentation_map)
    
    return pixel_area / dpcm ** 2


def new_image_size(old_image_size, image_size_bigger_side):
    if old_image_size[0] > old_image_size[1]:
        return image_size_bigger_side, int(image_size_bigger_side / old_image_size[0] * old_image_size[1])
    else:
        return int(image_size_bigger_side / old_image_size[1] * old_image_size[0]), image_size_bigger_side