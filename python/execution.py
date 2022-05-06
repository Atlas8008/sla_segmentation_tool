import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

from .plotting import combine_image_and_segmentation
from .segmentation import floodfill_segmentation, correct_holes, erase_small_components
from .utils import area_from_segmentation_map, new_image_size


def run_segmentation(
        dirname, 
        recursive, 
        dpi, 
        save_path, 
        border_crop_relative=0.02, 
        floodfill_tolerance=0.02, 
        image_size_bigger_side=1024,
        small_component_erase_percentage_threshold=2e-4,
        verbose=True,
    ):
    """Runs the segmentation algorithm over all images found in the provided folder and calculates the area via the pixel values.

    Args:
        dirname (str): The name of the directory containing the scans. Currently allowed image formats for the scans are jpeg, tiff and png.
        recursive (bool): Determines, if subdirectories should be included in the image file search.
            True - Also searches for images in subdirectories
            False - Only uses images from the specified dirname directory
        dpi (int): The DPI of the scans.
        save_path (str): The path where the result table will be saved. Should be a .csv file.
        border_crop_relative (float, optional): Before processing the image, parts of the border are cropped to prevent objects being detected at the image borders, which are usually irregular. This value designates the percentage to be cropped on each side. A value of 0.02 removes 2% of the image from the top, bottom, left and right of the image. Defaults to 0.02.
        floodfill_tolerance (float, optional): This value designates, how strongly neighboring pixels may differ to still be considered background during the initial segmentation. Higher values include more 'shadowy' pixels in the background, but may also include parts of the plants. The default values 0.01 should work for most cases, but feel free to change the values, if too little shadow or bigger parts of the plants are designated as background. Defaults to 0.02.
        image_size_bigger_side (int, optional): The image is being resized to increase processing speed at the cost of exactness of the result of the area. This value designates, to how many pixels the longer side of the image should be rescaled while keeping the aspect ratio of the image. The default value 1024 should be a good trade-off between speed and accuracy and should work well for most cases, while keeping the exactness error negligible. Try out different values at your own risk.. Defaults to 1024.
        small_component_erase_percentage_threshold (float, optional): This value is a threshold, determining if an component is too small to be kept in the segmentation. It is a value relative to the size of the complete image, i.e. if the value is 0.1 all object with a size smaller than 10% of the image are being removed. Defaults to 2e-4.
        verbose (bool, optional): If True, intermediate results and status updates will be prined. Defaults to True.

    Returns:
        pandas.DataFrame: The table containing the area information over all analyzed images.
    """
    
    image_regex = re.compile(r".*(JPE?G|TIFF?|PNG)")

    image_paths = []

    if recursive:
        for root, dirs, files in os.walk(dirname, topdown=True):
            for filename in files:
                if image_regex.search(filename.upper()):
                    image_paths.append(os.path.join(root, filename))
    else:
        for filename in os.listdir(dirname):
            if image_regex.search(filename.upper()):
                image_paths.append(os.path.join(dirname, filename))
                
    table_contents = []

    table_contents.append([
        "image_name", 
        "area_cm2", 
        "area_mm2", 
        "mean_area_cm2", 
        "mean_area_mm2", 
        "n_components", 
        "image_path"
    ])

    for img_nr, image_path in enumerate(image_paths):
        if verbose:
            print(f"{img_nr + 1}/{len(image_paths)} Image path {image_path}")
        
        image = Image.open(image_path).convert("RGB")
        
        if border_crop_relative != 0:
            ary = np.array(image)
            crop_h = int(border_crop_relative * ary.shape[0])
            crop_w = int(border_crop_relative * ary.shape[1])
            # Crop array
            ary = ary[
                crop_h:-crop_h,
                crop_w:-crop_w,
            ]
            image = Image.fromarray(ary)
                            
        old_size = image.size
        
        # Resize image
        new_size = new_image_size(image.size, image_size_bigger_side)
                            
        resize_ratio = max(new_size) / max(old_size)
        
        if verbose:
            print("Image resized by", resize_ratio)
        
        dpi_resized = resize_ratio * dpi
        
        image = image.resize(new_size, resample=Image.LANCZOS)
        image = np.array(image)
        
        if verbose:
            print(image.shape)
        
        segmentation = floodfill_segmentation(
            image, 
            tolerance=floodfill_tolerance, 
            verbose=True
        )
        segmentation, n_components = erase_small_components(
            segmentation, 
            min_image_percentage=small_component_erase_percentage_threshold
        )
        segmentation = correct_holes(
            image, 
            segmentation
        )
        
        plt.figure(figsize=(20, 20))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.imshow(segmentation)
        plt.show()
        
        combined = combine_image_and_segmentation(image, segmentation)
        
        plt.figure(figsize=(15, 15))
        plt.imshow(combined)
        plt.show()
                            
        area_cm2 = area_from_segmentation_map(segmentation, dpi_resized)
        
        table_contents.append([
            os.path.basename(image_path), 
            str(area_cm2), 
            str(area_cm2 * 100), 
            str(area_cm2 / n_components), 
            str(area_cm2 * 100 / n_components), 
            str(n_components),
            image_path
        ])
        
        if verbose:        
            print(f"Area: {area_cm2} cm²")
        
    if save_path:
        with open(save_path, "w") as f:
            table_rows = [",".join(row_contents) for row_contents in table_contents]
            
            f.write("\n".join(table_rows))
                
    df = pd.DataFrame(table_contents[1:], columns=table_contents[0])
        
    return df