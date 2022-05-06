import numpy as np

from scipy.ndimage import label


def floodfill_segmentation(image, starting_pixel=None, tolerance=0.04, verbose=False):    
    def pixel_average(image, pixel):
        return np.mean(image[pixel[1], pixel[0], :])
    
    if starting_pixel is None: # Retrieve corner pixel with maximal pixel average as starting point 
        possible_pixels = [(0, 0), (image.shape[1] - 1, 0), (0, image.shape[0] - 1), (image.shape[1] - 1, image.shape[0] - 1)]
        
        max_pixel = None
        
        for pixel in possible_pixels:
            if max_pixel is None or pixel_average(image, pixel) > pixel_average(image, max_pixel):
                max_pixel = pixel   
            
        starting_pixel = max_pixel
    
    pixels_visited_or_queued = set()
    next_pixels = [starting_pixel]  # FILO queue

    neighbor_indices = [
        np.array((0, -1)),
        np.array((0, 1)),
        np.array((1, 0)),
        np.array((-1, 0)),
    ]

    segmentation_map = np.ones_like(image[:, :, 0])

    pixel_count = 0
    
    last_text = ""

    while next_pixels:
        pixel_count += 1

        if pixel_count % 10000 == 0 and verbose:
            print("\b" * len(last_text), end="\r")
            last_text = "Pixel {}/{} ({}%)".format(pixel_count, np.prod(image.shape[:2]), int(pixel_count / np.prod(image.shape[:2]) * 100))
            print(last_text, end="")

        current_pixel = next_pixels.pop()

        pixels_visited_or_queued.add(current_pixel)

        current_pixel = np.array(current_pixel)

        segmentation_map[current_pixel[1], current_pixel[0]] = 0

        # Get mean pixel value
        avg_val = pixel_average(image, current_pixel)

        # Iterate neighboring pixels
        for n_index in neighbor_indices:
            next_pixel = tuple(current_pixel + n_index)

            #print(max_difference(image, current_pixel, next_pixel))

            if next_pixel not in pixels_visited_or_queued and \
               0 <= next_pixel[0] < image.shape[1] and \
               0 <= next_pixel[1] < image.shape[0] and \
               np.abs(avg_val - pixel_average(image, next_pixel)) < tolerance * 255: # * (0.75 * avg_val / 255.0 + 0.25):

                next_pixels.append(next_pixel)
                pixels_visited_or_queued.add(next_pixel)

    return segmentation_map


def get_component_indices(segmentation_map):
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    labeled, ncomponents = label(segmentation_map, structure)

    indices = np.indices(segmentation_map.shape).T[:, :, [1, 0]]
    indices = np.swapaxes(indices, 0, 1)
    
    component_indices_list = []

    for component_number in range(ncomponents):
        component_indices = indices[labeled == (component_number + 1)]
        
        component_indices_list.append(component_indices)
    
    return component_indices_list


def erase_small_components(segmentation_map, min_image_percentage=1e-4):
    component_indices = get_component_indices(segmentation_map)
    
    nr_erased = 0
    n_components = 0
    
    min_size = np.prod(segmentation_map.shape[:2]) * min_image_percentage
    
    for component in component_indices:
        # Test, if component is too small
        if len(component) < min_size:
            nr_erased += 1
            
            for idx in component:
                segmentation_map[idx[1], idx[0]] = 0
        else:
            print(f"Found component with {len(component)} pixels")
            n_components += 1
                
    print(f"\nErased {nr_erased} components that were too small (< {min_size} px)")
    
    return segmentation_map, n_components


def adjusted_2point_clustering(sample_list_means, samples):
    means =  []
    
    for stdev_sample in sample_list_means:
        means.append(np.mean(stdev_sample, axis=0))
    
    means = np.vstack(means)
    
    clusters = []
    
    for sample in samples:
        distances = np.linalg.norm((sample - means) * np.array([[0.75], [0.25]]), axis=1)
        
        if distances[0] < distances[1]:
            clusters.append(0)
        else:
            clusters.append(1)
            
    return np.array(clusters)


def correct_holes(image, segmentation_map):    
    segmentation_map_new = np.zeros_like(segmentation_map)
    
    clustered = adjusted_2point_clustering([image[segmentation_map == 0], image[segmentation_map == 1]], image[segmentation_map == 1])
        
    segmentation_map_new[segmentation_map == 1] = clustered
    
    return segmentation_map_new