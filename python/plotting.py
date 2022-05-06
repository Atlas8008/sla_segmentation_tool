import numpy as np


    
def combine_image_and_segmentation(img, segmentation):
    orig_seg = segmentation
    
    zero_seg = np.zeros_like(segmentation)
    segmentation = np.stack([segmentation, zero_seg, zero_seg], axis=-1)
    segmentation *= 255
    
    counter_segmentation = np.stack([zero_seg, zero_seg, 1 - orig_seg], axis=-1)
    counter_segmentation *= 255
    
    img_new = img * 0.6 + (segmentation + counter_segmentation) * 0.4
    
    return img_new.astype("uint8")