"""
## STATEMENT 1 : The number of objects in the scene is real : 11 , ans : 15 (Show image)
import cv2
def count_objects(image_path, min_area=900):
    # Load the image and convert it to grayscale
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC2.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to remove noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Threshold the image using Otsu's method
    _, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological opening and closing to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    opened_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel)
    closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)

    # Calculate connected components and their statistics
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(closed_image, connectivity=8)

    # Filter the connected components based on their area
    filtered_stats = [stat for stat in stats if min_area < stat[cv2.CC_STAT_AREA]]

    # Draw the detected objects on the image
    for stat, centroid in zip(stats, centroids):
        if min_area < stat[cv2.CC_STAT_AREA]:
            x, y, w, h = stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_TOP], stat[cv2.CC_STAT_WIDTH], stat[
                cv2.CC_STAT_HEIGHT]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(image, (int(centroid[0]), int(centroid[1])), 2, (0, 0, 255), -1)

    # Show the resulting image
    cv2.imshow('Detected Objects', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Return the number of objects (filtered connected components)
    return len(filtered_stats) - 1  # Subtract 1 to exclude the background component


# Example usage
image_path = 'NIVETHANRAMESH_TP062192_PIC2.jpg'
print("Number of objects:", count_objects(image_path))
"""

"""
## STATEMENT 2 : The number of objects of triangular shape are real : 0 , ans : 0 (show image)
import cv2

def count_triangular_objects(image_path, min_area=100):
    # Load the image and convert it to grayscale
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC2.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to remove noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Detect edges using the Canny edge detection method
    edged_image = cv2.Canny(blurred_image, 100, 150)

    # Apply morphological closing to fill small gaps in the contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed_image = cv2.morphologyEx(edged_image, cv2.MORPH_CLOSE, kernel)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a counter for triangular objects
    num_triangular_objects = 0

    for contour in contours:
        # Check the contour's area first to avoid unnecessary calculations
        if cv2.contourArea(contour) > min_area:
            # Approximate the contour to a simpler shape with a specified precision
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # If the approximated shape has 3 vertices, it's a triangle
            if len(approx) == 3:
                num_triangular_objects += 1
                cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)

    # Show the resulting image
    cv2.imshow('Detected Triangular Objects', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return num_triangular_objects

# Example usage
image_path = 'NIVETHANRAMESH_TP062192_PIC2.jpg'
print("Number of triangular objects:", count_triangular_objects(image_path))
"""

"""
## STATEMENT 3 : The smallest object in the scene is
import cv2

def find_smallest_object(image_path, min_area=100):
    # Load the image and convert it to grayscale
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC2.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to remove noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Detect edges using the Canny edge detection method
    edged_image = cv2.Canny(blurred_image, 10, 350)

    # Apply morphological closing to fill small gaps in the contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 12))
    closed_image = cv2.morphologyEx(edged_image, cv2.MORPH_CLOSE, kernel)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out contours with area smaller than min_area
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    if filtered_contours:
        # Find the contour with the smallest area
        smallest_contour = min(filtered_contours, key=cv2.contourArea)

        # Draw a bounding box around the smallest object
        x, y, w, h = cv2.boundingRect(smallest_contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save the resulting image
        cv2.imwrite('smallest_object_detected.jpg', image)
        cv2.imshow("smallest_object_detected.jpg", image)
        cv2.waitKey(0)

        return cv2.contourArea(smallest_contour), (x, y, w, h)
    else:
        return None, None

# Example usage
image_path = 'NIVETHANRAMESH_TP062192_PIC2.jpg'
smallest_area, smallest_object_coords = find_smallest_object(image_path)

if smallest_area is not None:
    print("Smallest object area:", smallest_area)
    print("Smallest object bounding box (x, y, width, height):", smallest_object_coords)
else:
    print("No objects detected.")
"""

"""
## STATEMENT 4 : The largest object in the scene is : X
import cv2

def find_largest_object(image_path, min_area=100):
    # Load the image and convert it to grayscale
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC2.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to remove noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Detect edges using the Canny edge detection method
    edged_image = cv2.Canny(blurred_image, 50, 70)

    # Apply morphological closing to fill small gaps in the contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    closed_image = cv2.morphologyEx(edged_image, cv2.MORPH_CLOSE, kernel)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out contours with area smaller than min_area
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    if filtered_contours:
        # Find the contour with the largest area
        largest_contour = max(filtered_contours, key=cv2.contourArea)

        # Draw a bounding box around the largest object
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save the resulting image
        cv2.imwrite('largest_object_detected.jpg', image)
        cv2.imshow('largest_object_detected.jpg', image)
        cv2.waitKey(0)

        return cv2.contourArea(largest_contour), (x, y, w, h)
    else:
        return None, None

# Example usage
image_path = 'NIVETHANRAMESH_TP062192_PIC2.jpg'
largest_area, largest_object_coords = find_largest_object(image_path)

if largest_area is not None:
    print("Largest object area:", largest_area)
    print("Largest object bounding box (x, y, width, height):", largest_object_coords)
else:
    print("No objects detected.")
"""

"""
## STATEMENT 5:The average size of the objects in the scene is: X (ASK SIR !!!!!!!!)
import cv2

def find_average_object_size(image_path, min_area=100):
    # Load the image and convert it to grayscale
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to remove noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Detect edges using the Canny edge detection method
    edged_image = cv2.Canny(blurred_image, 50, 150)

    # Apply morphological closing to fill small gaps in the contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed_image = cv2.morphologyEx(edged_image, cv2.MORPH_CLOSE, kernel)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out contours with area smaller than min_area
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    # Calculate the total area of the filtered objects
    total_area = sum(cv2.contourArea(c) for c in filtered_contours)

    # Calculate the average size of objects in the scene
    if filtered_contours:
        average_size = total_area / len(filtered_contours)
    else:
        average_size = 0

    # Draw the average object size as a square on the image
    avg_side_length = int(average_size ** 0.5)
    cv2.rectangle(image, (10, 10), (10 + avg_side_length, 10 + avg_side_length), (0, 255, 0), 2)

    # Save the resulting image
    cv2.imwrite('average_object_size.jpg', image)
    cv2.imshow('average_object_size.jpg', image)
    cv2.waitKey(0)

    return average_size

# Example usage
image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'
average_object_size = find_average_object_size(image_path)
print("Average object size:", average_object_size)
"""

"""
## STATEMENT 6 :The total area occupied by objects in the scene is: X
import cv2

def find_total_object_area(image_path, min_area=100):
    # Load the image and convert it to grayscale
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to remove noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Detect edges using the Canny edge detection method
    edged_image = cv2.Canny(blurred_image, 50, 150)

    # Apply morphological closing to fill small gaps in the contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed_image = cv2.morphologyEx(edged_image, cv2.MORPH_CLOSE, kernel)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out contours with area smaller than min_area
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    # Calculate the total area of the filtered objects using a list comprehension
    total_area = sum(cv2.contourArea(c) for c in filtered_contours)

    # Annotate the image with the total object area
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Total area occupied by objects: {total_area:.2f}"
    cv2.putText(image, text, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Save the resulting image
    cv2.imwrite('total_object_area.jpg', image)
    cv2.imshow('total_object_area.jpg', image)
    cv2.waitKey(0)

    return total_area

# Example usage
image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'
total_object_area = find_total_object_area(image_path)
print("Total area occupied by objects:", total_object_area)

"""
"""
## STATEMENT 7 :The percentage of the scene occupied by objects is: X%
import cv2

def find_percentage_occupied_by_objects(image_path, output_image_path, min_area=100, bilateral_diameter=5, bilateral_sigma_color=75, bilateral_sigma_space=75, block_size=15, c=-5, morph_iterations=2):
    # Load the image and convert it to grayscale
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a bilateral filter to reduce noise while preserving edges
    filtered_image = cv2.bilateralFilter(gray_image, bilateral_diameter, bilateral_sigma_color, bilateral_sigma_space)

    # Apply adaptive thresholding to segment the objects from the background
    thresholded_image = cv2.adaptiveThreshold(filtered_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c)

    # Apply morphological opening and closing operations to remove small noise and fill in small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
    closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)

    # Find and filter contours based on their area
    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    # Draw the filtered contours on the original image
    cv2.drawContours(image, filtered_contours, -1, (0, 255, 0), 2)

    # Save the processed image with highlighted objects
    cv2.imwrite('Percentage_of_sceneoccupied.jpg', image)
    cv2.imshow('Percentage_of_sceneoccupied.jpg', image)
    cv2.waitKey(0)

    # Calculate the total area of the filtered objects
    total_area = sum(cv2.contourArea(c) for c in filtered_contours)

    # Calculate the total area of the scene
    scene_area = gray_image.shape[0] * gray_image.shape[1]

    # Calculate the percentage of the scene occupied by objects
    percentage_occupied = (total_area / scene_area) * 100

    return percentage_occupied

# Example usage
image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'
output_image_path = 'Percentage_of_sceneoccupied.jpg'
percentage_occupied = find_percentage_occupied_by_objects(image_path, output_image_path)
print("Percentage of the scene occupied by objects:", percentage_occupied)
"""

"""
## STATEMENT 8 :The most common color among the objects in the scene is: X
import cv2
import numpy as np

def find_most_common_color(image_path, output_image_path, morph_iterations=2):
    # Load the image
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to segment the objects from the background
    thresholded_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, -5)

    # Apply morphological opening and closing operations to remove small noise and fill in small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
    closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)

    # Create a mask of the objects
    mask = cv2.bitwise_and(image, image, mask=closed_image)

    # Compute the average color of the segmented objects
    most_common_color = cv2.mean(mask, mask=closed_image)[:3]

    # Convert the most common color to integer values
    most_common_color = tuple(map(int, most_common_color))

    # Highlight the objects in the image with the most common color
    highlighted_image = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
    cv2.imwrite('Most_common_colour.jpg', highlighted_image)

    return most_common_color

# Example usage
image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'
output_image_path = 'Most_common_colour.jpg'
most_common_color = find_most_common_color(image_path, output_image_path)
print("Most common color among objects in the scene (BGR):", most_common_color)
"""

"""
## STATEMENT 9 :The number of objects with a specific color (e.g., red) is: X
import cv2
import numpy as np

def count_objects_with_color(image_path, output_image_path, lower_color, upper_color, morph_iterations=2):
    # Load the image
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color range and apply a mask
    mask = cv2.inRange(hsv_image, lower_color, upper_color)

    # Apply morphological opening and closing operations to remove small noise and fill in small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)

    # Find connected components in the mask
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(closed_mask)

    # Filter out small components (noise)
    min_area = 50
    object_count = sum([1 for stat in stats if stat[cv2.CC_STAT_AREA] > min_area]) - 1  # Subtract 1 to exclude the background component

    # Draw the detected components on the original image
    output_image = image.copy()
    for stat in stats:
        if stat[cv2.CC_STAT_AREA] > min_area:
            x, y, w, h = stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_TOP], stat[cv2.CC_STAT_WIDTH], stat[cv2.CC_STAT_HEIGHT]
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Highlight the red objects found in the output image
    highlighted_image = cv2.addWeighted(image, 0.7, cv2.bitwise_and(image, image, mask=closed_mask), 0.3, 0)

    # Save the output image
    cv2.imwrite(output_image_path, highlighted_image)
    cv2.imshow('Object_countwithcolour_red.jpg', highlighted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    return object_count

# Example usage
image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'
output_image_path = 'Object_countwithcolour_red.jpg'
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
object_count = count_objects_with_color(image_path, output_image_path, lower_red, upper_red)
print("Number of red objects in the scene:", object_count)
"""

"""
## STATEMENT 10 :The object with the highest aspect ratio is id: X
import cv2

def find_object_with_highest_aspect_ratio(image_path, morph_iterations=2, min_area=50, aspect_ratio_range=(1, 10)):
    # Load the image
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to segment the objects from the background
    thresholded_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, -5)

    # Apply morphological opening and closing operations to remove small noise and fill in small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
    closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)

    # Find contours in the closed image
    contours, _ = cv2.findContours(closed_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize the maximum aspect ratio and the corresponding contour
    max_aspect_ratio = 0
    max_aspect_ratio_contour = None

    # Iterate through contours and find the one with the highest aspect ratio
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = float(w) / h

        if area > min_area and aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1] and aspect_ratio > max_aspect_ratio:
            max_aspect_ratio = aspect_ratio
            max_aspect_ratio_contour = contour

    return max_aspect_ratio_contour, max_aspect_ratio

# Example usage
image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'
highest_aspect_ratio_contour, highest_aspect_ratio = find_object_with_highest_aspect_ratio(image_path)
print("Highest aspect ratio:", highest_aspect_ratio)
"""

"""
## STATEMENT 11 :The object with the lowest aspect ratio is id: X
import cv2

def find_object_with_lowest_aspect_ratio(image_path, morph_iterations=2, min_area=50, max_area=None, aspect_ratio_range=(1, 10)):
    # Load the image
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to segment the objects from the background
    thresholded_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, -5)

    # Calculate adaptive kernel size based on image dimensions
    kernel_size = (int(min(image.shape[:2]) * 0.03),) * 2

    # Apply morphological opening and closing operations to remove small noise and fill in small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    opened_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
    closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)

    # Find contours in the closed image
    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize the minimum aspect ratio and the corresponding contour
    min_aspect_ratio = float('inf')
    min_aspect_ratio_contour = None

    # Iterate through contours and find the one with the lowest aspect ratio
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = float(w) / h

        if min_area <= area and (max_area is None or area <= max_area) and aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1] and aspect_ratio < min_aspect_ratio:
            min_aspect_ratio = aspect_ratio
            min_aspect_ratio_contour = contour

    return min_aspect_ratio_contour, min_aspect_ratio

# Example usage
image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'
lowest_aspect_ratio_contour, lowest_aspect_ratio = find_object_with_lowest_aspect_ratio(image_path)
print("Lowest aspect ratio:", lowest_aspect_ratio)
"""

"""
## STATEMENT 12 :The object closest to the center of the scene is id: X
import cv2
import numpy as np

def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def find_object_closest_to_center(image_path, morph_iterations=2, blur_size=5):
    # Load the image
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (blur_size, blur_size), 0)

    # Apply adaptive thresholding to segment the objects from the background
    thresholded_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, -5)

    # Calculate adaptive kernel size based on image dimensions
    kernel_size = (int(min(image.shape[:2]) * 0.03),) * 2

    # Apply morphological opening and closing operations to remove small noise and fill in small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    opened_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
    closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)

    # Find contours in the closed image
    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the center of the image
    image_center = (image.shape[1] // 2, image.shape[0] // 2)

    # Initialize the minimum distance to the center and the corresponding contour
    min_distance = float('inf')
    closest_contour = None

    # Iterate through contours and find the one closest to the center
    for contour in contours:
        # Calculate the moments of the contour
        moments = cv2.moments(contour)

        # Calculate the center of the contour using moments
        if moments['m00'] != 0:
            contour_center = (moments['m10'] / moments['m00'], moments['m01'] / moments['m00'])

            # Calculate the distance between the contour center and the image center
            distance = euclidean_distance(image_center, contour_center)

            # Update the minimum distance and closest contour if needed
            if distance < min_distance:
                min_distance = distance
                closest_contour = contour

    return closest_contour, min_distance

# Example usage
image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'
closest_contour, closest_distance = find_object_closest_to_center(image_path)
print("Distance to the center:", closest_distance)
"""

"""
## STATEMENT 13 :The object farthest from the center of the scene is id: X
import cv2
import numpy as np

def euclidean_distance_sq(point1, point2):
    return (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2

def find_object_farthest_from_center(image_path, morph_iterations=2, blur_size=5, min_area_ratio=0.001):
    # Load the image
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (blur_size, blur_size), 0)

    # Apply Otsu's binarization to find the optimal threshold value
    _, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Calculate adaptive kernel size based on image dimensions
    kernel_size = (int(min(image.shape[:2]) * 0.03),) * 2

    # Apply morphological opening and closing operations to remove small noise and fill in small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    opened_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
    closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)

    # Find contours in the closed image
    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the center of the image
    image_center = (image.shape[1] // 2, image.shape[0] // 2)

    # Initialize the maximum distance to the center and the corresponding contour
    max_distance = float('-inf')
    farthest_contour = None

    # Define a minimum area threshold to filter out small contours
    min_area = min_area_ratio * image.shape[0] * image.shape[1]

    # Iterate through contours and find the one farthest from the center
    for contour in contours:
        area = cv2.contourArea(contour)

        if area > min_area:
            # Calculate the moments of the contour
            moments = cv2.moments(contour)

            # Calculate the center of the contour using moments
            if moments['m00'] != 0:
                contour_center = (moments['m10'] / moments['m00'], moments['m01'] / moments['m00'])

                # Calculate the squared distance between the contour center and the image center
                distance_sq = euclidean_distance_sq(image_center, contour_center)

                # Update the maximum distance and farthest contour if needed
                if distance_sq > max_distance:
                    max_distance = distance_sq
                    farthest_contour = contour

    return farthest_contour, np.sqrt(max_distance)

# Example usage
image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'
farthest_contour, farthest_distance = find_object_farthest_from_center(image_path)
print("Distance to the center from farthest:", farthest_distance)
"""

"""
## STATEMENT 14 :The most frequently occurring object type in the scene is: X (Continue here)
import cv2
import numpy as np

def classify_shape(contour):
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    num_vertices = len(approx)

    if num_vertices == 3:
        return 'triangle'
    elif num_vertices == 4:
        _, _, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if 0.9 <= aspect_ratio <= 1.1:
            return 'square'
        else:
            return 'rectangle'
    elif num_vertices > 4:
        return 'circle'
    else:
        return 'unknown'


def filter_non_convex(contours):
    filtered_contours = []
    for contour in contours:
        hull = cv2.convexHull(contour)
        if cv2.contourArea(hull) / cv2.contourArea(contour) < 1.2:
            filtered_contours.append(contour)
    return filtered_contours


def find_most_frequent_object_type(image_path, morph_iterations=2, min_contour_area_ratio=0.001):
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresholded_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                              31, -15)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
    closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)

    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = filter_non_convex(contours)

    shape_counter = {}

    min_contour_area = min_contour_area_ratio * image.shape[0] * image.shape[1]

    for contour in contours:
        if cv2.contourArea(contour) >= min_contour_area:
            shape = classify_shape(contour)
            if shape in shape_counter:
                shape_counter[shape] += 1
            else:
                shape_counter[shape] = 1

            color = (0, 255, 0)
            thickness = 2
            cv2.drawContours(image, [contour], 0, color, thickness)

    most_frequent_shape = max(shape_counter, key=shape_counter.get)

    return most_frequent_shape, shape_counter, image

image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'
most_frequent_shape, shape_counter, image_with_shapes = find_most_frequent_object_type(image_path)

print("The most frequently occurring object type is:", most_frequent_shape)
print("Object type counts:", shape_counter)

cv2.imshow('Detected Shapes', image_with_shapes)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
## STATEMENT 15 :The number of objects partially occluded by other objects is: X
import cv2
import numpy as np

def find_occluded_objects(image_path):
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Use Canny edge detection
    edges = cv2.Canny(blurred_image, 50, 150)

    # Apply morphological operations to reduce complexity
    kernel = np.ones((3, 3), np.uint8)
    morphed_image = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(morphed_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    num_occluded_objects = 0

    if hierarchy is not None:
        for i in range(len(contours)):
            # If the contour has a parent, it is considered occluded
            if hierarchy[0][i][3] != -1:
                num_occluded_objects += 1

                # Draw contours of occluded objects
                cv2.drawContours(image, contours, i, (0, 255, 0), 2)

    return num_occluded_objects, image

image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'
num_occluded_objects, image_with_objects = find_occluded_objects(image_path)

print("The number of partially occluded objects is:", num_occluded_objects)

cv2.imshow('Detected Objects', image_with_objects)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
## STATEMENT 16 :The number of objects touching the border of the image is: X
import cv2
import numpy as np

def find_objects_touching_border(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)

    # Use Canny edge detection
    edged_image = cv2.Canny(blurred_image, 50, 200)

    # Dilation operation to connect edges
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(edged_image, kernel, iterations=2)

    # Erosion operation to remove small noise after dilation
    eroded_image = cv2.erode(dilated_image, kernel, iterations=1)

    # Additional morphological operations to better connect edges and reduce noise
    morphed_image = cv2.morphologyEx(eroded_image, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(morphed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    num_objects_touching_border = 0
    image_height, image_width = gray_image.shape

    for contour in contours:
        for point in contour:
            x, y = point[0][0], point[0][1]
            if x == 0 or y == 0 or x == image_width - 1 or y == image_height - 1:
                num_objects_touching_border += 1
                cv2.drawContours(image, [contour], 0, (0, 0, 255), 2)  # Draw in red
                break

    return num_objects_touching_border, image

image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'
num_objects_touching_border, image_with_border_objects = find_objects_touching_border(image_path)

print("The number of objects touching the border of the image is:", num_objects_touching_border)

cv2.imshow('Objects Touching Border', image_with_border_objects)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
## STATEMENT 17 :The average distance between objects in the scene is: X
import cv2
import numpy as np

def find_average_distance_between_objects(image_path):
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)

    thresholded_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, -7)

    kernel = np.ones((3, 3), np.uint8)
    morphed_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel, iterations=2)
    morphed_image = cv2.morphologyEx(morphed_image, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(morphed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Filter out small contours and noise
            M = cv2.moments(contour)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))

    num_centroids = len(centroids)
    distances = []
    for i in range(num_centroids):
        for j in range(i + 1, num_centroids):
            distance = np.sqrt((centroids[i][0] - centroids[j][0]) ** 2 + (centroids[i][1] - centroids[j][1]) ** 2)
            distances.append(distance)

    if distances:
        average_distance = np.median(distances)  # Use median instead of mean to reduce the effect of outliers
    else:
        average_distance = 0

    return average_distance

image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'
average_distance = find_average_distance_between_objects(image_path)

print("The average distance between objects in the scene is:", average_distance)
"""

"""
## STATEMENT 18 :The total perimeter of all objects in the scene is: X
import cv2
import numpy as np

def find_total_perimeter(image_path):
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)

    thresholded_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, -7)

    kernel = np.ones((3, 3), np.uint8)
    morphed_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel, iterations=2)
    morphed_image = cv2.morphologyEx(morphed_image, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(morphed_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    total_perimeter = 0
    min_perimeter = 30  # Filter out small contours and noise, adjusted for nested contours
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter > min_perimeter:
            total_perimeter += perimeter

    return total_perimeter

image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'
total_perimeter = find_total_perimeter(image_path)

print("The total perimeter of all objects in the scene is:", total_perimeter)
"""

"""
## STATEMENT 19 :The average perimeter of the objects in the scene is: X
import cv2
import numpy as np

def find_average_perimeter(image_path):
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)

    thresholded_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, -7)

    kernel = np.ones((5, 5), np.uint8)
    morphed_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel, iterations=2)
    morphed_image = cv2.morphologyEx(morphed_image, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(morphed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    total_perimeter = 0
    min_perimeter = 30  # Filter out small contours and noise
    num_valid_contours = 0
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter > min_perimeter:
            total_perimeter += perimeter
            num_valid_contours += 1

    if num_valid_contours == 0:
        return 0
    else:
        return total_perimeter / num_valid_contours

image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'
average_perimeter = find_average_perimeter(image_path)

print("The average perimeter of the objects in the scene is:", average_perimeter)
"""

"""
## STATEMENT 20 :The object with the highest number of neighbors (other objects close to it) is id: X
import cv2
import numpy as np

def distance_matrix(points):
    num_points = len(points)
    matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(i + 1, num_points):
            dist = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
            matrix[i, j] = dist
            matrix[j, i] = dist
    return matrix

def preprocess_image(image_path):
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    thresholded_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, -7)
    kernel = np.ones((3, 3), np.uint8)
    morphed_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel, iterations=2)
    morphed_image = cv2.morphologyEx(morphed_image, cv2.MORPH_OPEN, kernel, iterations=2)
    return morphed_image

def get_largest_contour(contours):
    max_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour
    return largest_contour

def find_object_with_most_neighbors(image_path, distance_threshold_ratio=0.15):
    morphed_image = preprocess_image('NIVETHANRAMESH_TP062192_PIC1.jpg')
    contours, _ = cv2.findContours(morphed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 100
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]

    centroids = []
    for contour in filtered_contours:
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        centroids.append((cx, cy))

    avg_area = np.mean([cv2.contourArea(contour) for contour in filtered_contours])
    distance_threshold = distance_threshold_ratio * np.sqrt(avg_area)
    dist_matrix = distance_matrix(centroids)

    max_neighbors = -1
    max_neighbor_objects = []
    for i, centroid in enumerate(centroids):
        neighbor_count = sum(1 for j, other_centroid in enumerate(centroids) if i != j and dist_matrix[i, j] <= distance_threshold)
        if neighbor_count > max_neighbors:
            max_neighbors = neighbor_count
            max_neighbor_objects = [filtered_contours[i]]
        elif neighbor_count == max_neighbors:
            max_neighbor_objects.append(filtered_contours[i])

    max_neighbor_object = get_largest_contour(max_neighbor_objects)

    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')
    output_image = image.copy()
    cv2.drawContours(output_image, [max_neighbor_object], -1, (0, 255, 0), 3)

    # Show the output image
    cv2.imshow('Detected Object', output_image)
    cv2.waitKey(0)

    return max_neighbor_object, max_neighbors

image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'
distance_threshold_ratio = 0.15
object_with_most_neighbors, num_neighbors = find_object_with_most_neighbors(image_path, distance_threshold_ratio)

print("The object with the highest number of neighbors has", num_neighbors, "neighbors.")
"""

"""
## STATEMENT 21 :The object with the lowest number of neighbors (other objects close to it) is id: X
import cv2
import numpy as np

def distance_matrix(points):
    num_points = len(points)
    matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(i + 1, num_points):
            dist = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
            matrix[i, j] = dist
            matrix[j, i] = dist
    return matrix

def find_object_with_least_neighbors(image_path, distance_threshold_ratio=0.2):
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    _, binary_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours based on area
    min_area = 100
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]

    # Compute centroids of filtered contours
    centroids = []
    for contour in filtered_contours:
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        centroids.append((cx, cy))

    # Calculate average contour area and standard deviation
    areas = [cv2.contourArea(contour) for contour in filtered_contours]
    avg_area = np.mean(areas)
    std_area = np.std(areas)

    # Update distance threshold based on average area and standard deviation
    distance_threshold = distance_threshold_ratio * (avg_area + std_area)

    dist_matrix = distance_matrix(centroids)

    min_neighbors = float('inf')
    min_neighbor_object = None
    for i, centroid in enumerate(centroids):
        neighbor_count = sum(1 for j, other_centroid in enumerate(centroids) if i != j and dist_matrix[i, j] <= distance_threshold)
        if neighbor_count < min_neighbors:
            min_neighbors = neighbor_count
            min_neighbor_object = filtered_contours[i]

    # Draw the detected object on the original image
    output_image = image.copy()
    cv2.drawContours(output_image, [min_neighbor_object], -1, (0, 255, 0), 3)

    # Show the output image
    cv2.imshow('Detected Object', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return min_neighbor_object, min_neighbors

image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'
distance_threshold_ratio = 0.2
object_with_least_neighbors, num_neighbors = find_object_with_least_neighbors(image_path, distance_threshold_ratio)

print("The object with the lowest number of neighbors has", num_neighbors, "neighbors.")
"""

"""
## STATEMENT 22 :The most common spatial relationship between objects (e.g., above, below, left, right) is: X
import cv2
import numpy as np

def find_most_common_relationship(image_path):
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    _, _, stats, centroids = cv2.connectedComponentsWithStats(opening)

    relationship_count = {'right': 0, 'left': 0, 'below': 0, 'above': 0}
    for i, centroid in enumerate(centroids):
        for other_centroid in centroids:
            if (centroid == other_centroid).all():
                continue

            dx = other_centroid[0] - centroid[0]
            dy = other_centroid[1] - centroid[1]

            if abs(dx) > abs(dy):
                if dx > 0:
                    relationship_count['right'] += 1
                else:
                    relationship_count['left'] += 1
            else:
                if dy > 0:
                    relationship_count['below'] += 1
                else:
                    relationship_count['above'] += 1

    most_common_relationship = max(relationship_count, key=relationship_count.get)
    return most_common_relationship, relationship_count[most_common_relationship]

image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'
most_common_relationship, count = find_most_common_relationship(image_path)

print("The most common spatial relationship is", most_common_relationship, "with", count, "occurrences.")
"""

"""
## STATEMENT 23 :The number of groups/clusters of objects in the scene is: X  (The code is not working. Check again)
import cv2
import numpy as np

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def agglomerative_clustering(centroids, distance_threshold):
    clusters = [[centroid] for centroid in centroids]

    while True:
        min_distance = float('inf')
        merge_pair = None

        for i, cluster1 in enumerate(clusters[:-1]):
            for j, cluster2 in enumerate(clusters[i + 1:]):
                for point1 in cluster1:
                    for point2 in cluster2:
                        distance = euclidean_distance(point1, point2)
                        if distance < min_distance:
                            min_distance = distance
                            merge_pair = (i, i + 1 + j)

        if min_distance > distance_threshold or merge_pair is None:
            break

        clusters[merge_pair[0]] += clusters[merge_pair[1]]
        del clusters[merge_pair[1]]

    return clusters

def find_clusters(image_path, distance_threshold):
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    _, _, stats, centroids = cv2.connectedComponentsWithStats(opening)

    clusters = agglomerative_clustering(centroids, distance_threshold)

    # Draw clusters on the image
    output_image = image.copy()
    for i, centroid in enumerate(centroids):
        color = (0, 255, 0)  # Green color for clustered points
        cv2.circle(output_image, (int(centroid[0]), int(centroid[1])), 5, color, -1)

    # Show the output image
    cv2.imshow('Detected Clusters', output_image)
    cv2.waitKey(0)

    return len(clusters)

image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'
distance_threshold = 50
num_clusters = find_clusters(image_path, distance_threshold)

print("The number of groups/clusters of objects in the scene is:", num_clusters)
"""

"""
## STATEMENT 24 :The largest group/cluster of objects in the scene has X objects
import cv2
import numpy as np

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return

        if self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_x] = root_y
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_y] += 1

def union_find_clustering(centroids, distance_threshold):
    n = len(centroids)
    uf = UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            if euclidean_distance(centroids[i], centroids[j]) < distance_threshold:
                uf.union(i, j)

    clusters = {}
    for i in range(n):
        root = uf.find(i)
        if root in clusters:
            clusters[root].append(centroids[i])
        else:
            clusters[root] = [centroids[i]]

    return list(clusters.values())

def find_largest_cluster(image_path, distance_threshold):
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    _, _, _, centroids = cv2.connectedComponentsWithStats(opening)
    centroids = centroids[1:].tolist()

    clusters = union_find_clustering(centroids, distance_threshold)

    largest_cluster = max(clusters, key=len)

    output_image = image.copy()
    for centroid in largest_cluster:
        color = (0, 255, 0)
        cv2.circle(output_image, (int(centroid[0]), int(centroid[1])), 5, color, -1)

    cv2.imshow('Detected Largest Cluster', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return len(largest_cluster)

image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'
distance_threshold = 50
num_objects_in_largest_cluster = find_largest_cluster(image_path, distance_threshold)

print("The largest group/cluster of objects in the scene has", num_objects_in_largest_cluster, "objects.")
"""

"""
## STATEMENT 25 :The smallest group/cluster of objects in the scene has X objects
import cv2
import numpy as np

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def region_query(centroids, point_idx, distance_threshold):
    neighbors = []
    for i, centroid in enumerate(centroids):
        if euclidean_distance(centroids[point_idx], centroid) < distance_threshold:
            neighbors.append(i)
    return neighbors

def dbscan(centroids, distance_threshold, min_points):
    labels = [None] * len(centroids)
    cluster_id = 0

    for point_idx in range(len(centroids)):
        if labels[point_idx] is not None:
            continue

        neighbors = region_query(centroids, point_idx, distance_threshold)
        if len(neighbors) < min_points:
            labels[point_idx] = -1
        else:
            labels[point_idx] = cluster_id
            to_visit = neighbors[:]
            for neighbor_idx in to_visit:
                if labels[neighbor_idx] is None:
                    labels[neighbor_idx] = cluster_id
                    new_neighbors = region_query(centroids, neighbor_idx, distance_threshold)
                    if len(new_neighbors) >= min_points:
                        to_visit.extend(new_neighbors)

            cluster_id += 1

    clusters = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue
        if label in clusters:
            clusters[label].append(centroids[idx])
        else:
            clusters[label] = [centroids[idx]]

    return list(clusters.values())

def find_smallest_cluster(image_path, distance_threshold, min_points):
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.medianBlur(gray_image, 5)
    binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    _, _, _, centroids = cv2.connectedComponentsWithStats(opening)
    centroids = centroids[1:].tolist()

    clusters = dbscan(centroids, distance_threshold, min_points)

    smallest_cluster = min(clusters, key=len)

    output_image = image.copy()
    for centroid in smallest_cluster:
        color = (0, 0, 255)  # Red color for the smallest cluster
        cv2.circle(output_image, (int(centroid[0]), int(centroid[1])), 5, color, -1)

    cv2.imshow('Detected Smallest Cluster', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return len(smallest_cluster)

image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'
distance_threshold = 50
min_points = 3
num_objects_in_smallest_cluster = find_smallest_cluster(image_path, distance_threshold, min_points)

print("The smallest group/cluster of objects in the scene has", num_objects_in_smallest_cluster, "objects.")
"""

"""
## STATEMENT 26 :The average number of objects per group/cluster in the scene is: X
import cv2
import numpy as np

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def nearest_neighbor_clustering(centroids, distance_threshold):
    clusters = []

    while centroids:
        centroid = centroids.pop(0)
        cluster = [centroid]

        for other_centroid in centroids.copy():
            if any(euclidean_distance(c, other_centroid) < distance_threshold for c in cluster):
                cluster.append(other_centroid)
                centroids.remove(other_centroid)

        clusters.append(cluster)

    return clusters

def average_objects_per_cluster(image_path, distance_threshold):
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.medianBlur(gray_image, 5)

    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    for contour in contours:
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        centroids.append((cx, cy))

    clusters = nearest_neighbor_clustering(centroids, distance_threshold)
    num_clusters = len(clusters)

    if num_clusters == 0:
        return 0

    total_objects = sum(len(cluster) for cluster in clusters)
    average_objects = total_objects / num_clusters

    # Draw clusters on output image
    output_image = image.copy()
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    for i, cluster in enumerate(clusters):
        color = colors[i % len(colors)]

        for centroid in cluster:
            cv2.circle(output_image, (int(centroid[0]), int(centroid[1])), 5, color, -1)

    cv2.imshow('Detected Objects and Clusters', output_image)
    cv2.waitKey(0)

    return average_objects

image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'
distance_threshold = 50
average_objects_per_cluster = average_objects_per_cluster(image_path, distance_threshold)

print("The average number of objects per group/cluster in the scene is", average_objects_per_cluster)
"""

"""
## STATEMENT 27 :The number of objects with a specific orientation (e.g., horizontal, vertical, or diagonal) is: X
import cv2
import numpy as np

def get_orientation(angle):
    if -45 < angle <= 45 or 135 < angle <= 180 or -180 < angle <= -135:
        return 'horizontal'
    elif -135 < angle <= -45 or 45 < angle <= 135:
        return 'vertical'
    else:
        return 'diagonal'

def count_objects_by_orientation(image_path, orientation):
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, binary_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    output_image = image.copy()

    for contour in contours:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            angle = ellipse[2]

            if get_orientation(angle) == orientation:
                count += 1
                cv2.drawContours(output_image, [contour], 0, (0, 255, 0), 2)

    cv2.imshow(f'{orientation.capitalize()} Objects', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return count

image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'
orientation = 'horizontal'  # Change this to 'vertical' or 'diagonal' as needed
count = count_objects_by_orientation(image_path, orientation)

print(f"The number of {orientation} objects in the scene is", count)
"""

"""
## STATEMENT 28 :The average orientation of the objects in the scene is: X
import cv2
import numpy as np

def average_orientation(image_path):
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 100, 200)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

    angles = []
    output_image = image.copy()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)

            cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Objects in the Scene', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(angles) > 0:
        avg_angle = np.mean(angles)
    else:
        avg_angle = 0

    return avg_angle

image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'
avg_angle = average_orientation(image_path)

print(f"The average orientation of the objects in the scene is {avg_angle:.2f} degrees.")
"""

"""
## STATEMENT 29 :The object with the highest color intensity is id: X
import cv2
import numpy as np

def highest_color_intensity_object(image_path):
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.medianBlur(gray_image, 5)
    _, binary_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY_INV)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    max_intensity = 0
    max_intensity_label = None
    max_intensity_contour = None

    for label in range(1, num_labels):
        mask = np.where(labels == label, 255, 0).astype(np.uint8)
        masked_image = cv2.bitwise_and(lab_image, lab_image, mask=mask)
        intensity = np.sum(masked_image)
        if intensity > max_intensity:
            max_intensity = intensity
            max_intensity_label = label

            # Finding the corresponding contour for the label
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_intensity_contour = contours[0]

    return max_intensity_contour, max_intensity_label

image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'
contour, contour_id = highest_color_intensity_object(image_path)

if contour is not None:
    image = cv2.imread(image_path)
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 3)
    cv2.putText(image, f"ID: {contour_id}", tuple(contour[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Object with Highest Color Intensity", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No objects found in the image.")
"""

"""
## STATEMENT 30 :The object with the lowest color intensity is id: X
import cv2
import numpy as np

def lowest_color_intensity_object(image_path):
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.medianBlur(gray_image, 5)
    _, binary_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY_INV)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    min_intensity = float('inf')
    min_intensity_label = None
    min_intensity_contour = None

    for label in range(1, num_labels):
        mask = np.where(labels == label, 255, 0).astype(np.uint8)
        masked_image = cv2.bitwise_and(lab_image, lab_image, mask=mask)
        intensity_sum = np.sum(masked_image[:, :, 0])
        area = np.count_nonzero(mask)
        mean_intensity = intensity_sum / area if area > 0 else float('inf')

        if mean_intensity < min_intensity:
            min_intensity = mean_intensity
            min_intensity_label = label

            # Finding the corresponding contour for the label
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_intensity_contour = contours[0]

    return min_intensity_contour, min_intensity_label

image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'
contour, contour_id = lowest_color_intensity_object(image_path)

if contour is not None:
    image = cv2.imread(image_path)
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 3)
    cv2.putText(image, f"ID: {contour_id}", tuple(contour[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Object with Lowest Color Intensity", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No objects found in the image.")
"""

"""
## STATEMENT 31 :The average color intensity of the objects in the scene is: X
import cv2
import numpy as np

def average_color_intensity(image_path):
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.medianBlur(gray_image, 5)
    _, binary_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY_INV)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    total_intensity = 0
    total_objects = num_labels - 1

    for label in range(1, num_labels):
        mask = np.where(labels == label, 255, 0).astype(np.uint8)
        masked_image = cv2.bitwise_and(lab_image, lab_image, mask=mask)
        intensity = np.sum(masked_image[:, :, 0])
        total_intensity += intensity

        # Draw contour and display ID
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
        cv2.putText(image, f"ID: {label}", tuple(contours[0][0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if total_objects > 0:
        average_intensity = total_intensity / total_objects
    else:
        average_intensity = 0

    return average_intensity, image

image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'
average_intensity, image = average_color_intensity(image_path)

cv2.putText(image, f"Average Intensity: {average_intensity:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imshow("Average Color Intensity", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
## STATEMENT 32 :The number of objects with a specific level of brightness (e.g., low brightness, high brightness) is: X
import cv2
import numpy as np

def count_objects_by_brightness(image_path, brightness_threshold):
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.medianBlur(gray_image, 5)
    _, binary_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY_INV)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    low_brightness_objects = 0
    high_brightness_objects = 0

    for label in range(1, num_labels):
        mask = np.where(labels == label, 255, 0).astype(np.uint8)
        masked_image = cv2.bitwise_and(lab_image, lab_image, mask=mask)
        intensity_sum = np.sum(masked_image[:, :, 0])
        area = np.count_nonzero(mask)
        brightness = intensity_sum / area if area > 0 else 0

        if brightness < brightness_threshold:
            low_brightness_objects += 1
            contour_color = (0, 0, 255)  # Red
        else:
            high_brightness_objects += 1
            contour_color = (0, 255, 0)  # Green

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, contour_color, 2)

    return low_brightness_objects, high_brightness_objects, image

image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'
brightness_threshold = 100  # Adjust this value based on your requirements

low_brightness_objects, high_brightness_objects, image = count_objects_by_brightness(image_path, brightness_threshold)

print(f"Number of low brightness objects: {low_brightness_objects}")
print(f"Number of high brightness objects: {high_brightness_objects}")

cv2.imshow("Objects by Brightness", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
## STATEMENT 33 :The average brightness of the objects in the scene is: X
import cv2
import numpy as np

def average_brightness_of_objects(image_path):
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.medianBlur(gray_image, 5)
    _, binary_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY_INV)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    total_brightness = 0
    total_objects = num_labels - 1

    for label in range(1, num_labels):
        mask = np.where(labels == label, 255, 0).astype(np.uint8)
        masked_image = cv2.bitwise_and(lab_image, lab_image, mask=mask)
        intensity_sum = np.sum(masked_image[:, :, 0])
        area = np.count_nonzero(mask)
        if area > 0:
            brightness = intensity_sum * (1 / area) # Multiplying is faster than division
            total_brightness += brightness

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    average_brightness = total_brightness / total_objects if total_objects > 0 else 0

    return average_brightness, image

image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'

average_brightness, image = average_brightness_of_objects(image_path)

print(f"Average brightness of objects: {average_brightness}")

cv2.imshow("Average Brightness of Objects", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
## STATEMENT 34 :The number of objects with a specific level of contrast with their surroundings is: X
import cv2
import numpy as np

def count_objects_with_contrast(image_path, contrast_threshold=30):
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.medianBlur(gray_image, 5)
    binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    count = 0

    for label in range(1, num_labels):
        mask = np.where(labels == label, 255, 0).astype(np.uint8)

        masked_image = cv2.bitwise_and(gray_image, gray_image, mask=mask)
        surrounding_mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
        surrounding_mask = cv2.subtract(surrounding_mask, mask)

        surrounding_image = cv2.bitwise_and(gray_image, gray_image, mask=surrounding_mask)

        object_brightness = np.mean(masked_image[mask > 0])
        surrounding_brightness = np.mean(surrounding_image[surrounding_mask > 0])

        contrast = abs(object_brightness - surrounding_brightness)

        if contrast >= contrast_threshold:
            count += 1
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    return count, image

image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'
contrast_threshold = 30

object_count, image = count_objects_with_contrast(image_path, contrast_threshold)

print(f"Number of objects with contrast >= {contrast_threshold}: {object_count}")

cv2.imshow("Objects with Contrast", image)
cv2.waitKey(0)
"""

"""
## STATEMENT 35 :The object with the highest contrast with its surroundings is id: X
import cv2
import numpy as np

def object_with_highest_contrast(image_path):
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.medianBlur(gray_image, 5)
    binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    edges = cv2.Canny(binary_image, 100, 200)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity=8)

    max_contrast = 0
    max_contrast_label = None

    dx = cv2.Scharr(gray_image, cv2.CV_32F, 1, 0)
    dy = cv2.Scharr(gray_image, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(dx, dy)

    for label in range(1, num_labels):
        mask = np.where(labels == label, 255, 0).astype(np.uint8)

        masked_magnitude = cv2.bitwise_and(magnitude, magnitude, mask=mask)
        surrounding_mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
        surrounding_mask = cv2.subtract(surrounding_mask, mask)

        surrounding_magnitude = cv2.bitwise_and(magnitude, magnitude, mask=surrounding_mask)

        object_contrast = np.mean(masked_magnitude[mask > 0])
        surrounding_contrast = np.mean(surrounding_magnitude[surrounding_mask > 0])

        contrast = abs(object_contrast - surrounding_contrast)

        if contrast > max_contrast:
            max_contrast = contrast
            max_contrast_label = label

    mask = np.where(labels == max_contrast_label, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    return max_contrast_label, max_contrast, image

image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'

highest_contrast_object_id, highest_contrast, image = object_with_highest_contrast(image_path)

print(f"The object with the highest contrast (id: {highest_contrast_object_id}) has a contrast of {highest_contrast}")

cv2.imshow("Highest Contrast Object", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
## STATEMENT 36 :The number of objects with a specific hue range (e.g., between 0 and 60) is: X
import cv2
import numpy as np

def count_objects_with_hue_range(image_path, hue_lower, hue_upper):
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_hue_range = np.array([hue_lower, 50, 50])
    upper_hue_range = np.array([hue_upper, 255, 255])

    hue_mask = cv2.inRange(hsv_image, lower_hue_range, upper_hue_range)

    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morphed_mask = cv2.morphologyEx(hue_mask, cv2.MORPH_CLOSE, kernel)
    morphed_mask = cv2.morphologyEx(morphed_mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(morphed_mask, connectivity=8)

    # Draw bounding boxes for the objects
    for stat in stats[1:]:
        x, y, w, h, _ = stat
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return num_labels - 1, image

image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'
hue_lower = 0
hue_upper = 60

num_objects_in_hue_range, image = count_objects_with_hue_range(image_path, hue_lower, hue_upper)

print(f"The number of objects with hue range {hue_lower}-{hue_upper} is {num_objects_in_hue_range}")

cv2.imshow("Objects in Hue Range", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
## STATEMENT 37 :The number of objects with a specific aspect ratio range (e.g., between 1 and 2) is: X
import cv2
import numpy as np

def count_objects_with_aspect_ratio_range(image_path, min_aspect_ratio, max_aspect_ratio, min_area=100):
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.medianBlur(gray_image, 5)
    binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened_image, connectivity=8)

    object_count = 0

    for stat in stats[1:]:
        x, y, w, h, area = stat
        aspect_ratio = float(w) / float(h)

        if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio and area > min_area:
            object_count += 1
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return object_count, image

image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'
min_aspect_ratio = 1
max_aspect_ratio = 2

num_objects_in_aspect_ratio_range, image = count_objects_with_aspect_ratio_range(image_path, min_aspect_ratio, max_aspect_ratio)

print(f"The number of objects with aspect ratio range {min_aspect_ratio}-{max_aspect_ratio} is {num_objects_in_aspect_ratio_range}")

cv2.imshow("Objects with Aspect Ratio Range", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
## STATEMENT 38 :The object with the highest number of corners/vertices is id: X
import cv2
import numpy as np

def find_object_with_most_corners(image_path):
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.medianBlur(gray_image, 5)
    binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_corners = 0
    max_corners_object = None

    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True) # Adjust the epsilon value for better corner detection
        approx_corners = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx_corners) > max_corners:
            max_corners = len(approx_corners)
            max_corners_object = contour

    cv2.drawContours(image, [max_corners_object], -1, (0, 255, 0), 2)

    return max_corners, image

image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'

max_corners, image = find_object_with_most_corners(image_path)

print(f"The object with the highest number of corners/vertices has {max_corners} corners.")

cv2.imshow("Object with Most Corners", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
## STATEMENT 39 :The number of objects with a specific saturation range (e.g., between 0.3 and 0.6) is: X
import cv2
import numpy as np

def count_objects_with_saturation_range(image_path, lower_saturation, upper_saturation):
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_range = (0, int(lower_saturation * 255), 0)
    upper_range = (179, int(upper_saturation * 255), 255)

    mask = cv2.inRange(hsv_image, lower_range, upper_range)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)

    object_count = 0
    for i, stat in enumerate(stats[1:]):
        x, y, w, h, area = stat
        object_count += 1
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return object_count, image

image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'
lower_saturation = 0.3
upper_saturation = 0.6

num_objects, image = count_objects_with_saturation_range(image_path, lower_saturation, upper_saturation)

print(f"There are {num_objects} objects with a saturation range between {lower_saturation} and {upper_saturation}.")

cv2.imshow("Objects with Saturation Range", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
## STATEMENT 40 :The average number of corners/vertices per object in the scene is: X
import cv2
import numpy as np

def find_average_corners_per_object(image_path):
    image = cv2.imread('NIVETHANRAMESH_TP062192_PIC1.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.medianBlur(gray_image, 5)
    binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_corners = 0
    object_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        epsilon = 0.01 * np.sqrt(area)  # Adaptively calculate epsilon based on contour area
        approx_corners = cv2.approxPolyDP(contour, epsilon, True)

        total_corners += len(approx_corners)
        object_count += 1

        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
        for corner in approx_corners:
            x, y = corner.ravel()
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

    average_corners = total_corners / object_count if object_count > 0 else 0

    return average_corners, image

image_path = 'NIVETHANRAMESH_TP062192_PIC1.jpg'
average_corners, image = find_average_corners_per_object(image_path)

print("Average number of corners per object:", average_corners)

cv2.imshow("Objects and Corners", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

## Statement 41: Implement an unsupervised learning approach to cluster objects based on their visual properties

import cv2
import numpy as np

def extract_object_features(image_path, min_area=100):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edged_image = cv2.Canny(blurred_image, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,4))
    closed_image = cv2.morphologyEx(edged_image, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    object_features = []
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = float(w) / h
            object_features.append([area, aspect_ratio])

    return np.array(object_features), contours

def initialize_centroids(data, k):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    return centroids

def assign_clusters(data, centroids):
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    cluster_assignments = np.argmin(distances, axis=0)
    return cluster_assignments

def update_centroids(data, cluster_assignments, k):
    new_centroids = np.array([data[cluster_assignments == i].mean(axis=0) for i in range(k)])
    return new_centroids

def kmeans_clustering(data, k, max_iterations=100):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iterations):
        cluster_assignments = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, cluster_assignments, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, cluster_assignments

def main(image_path, k=2):
    object_features, contours = extract_object_features(image_path)
    filtered_contours = [c for c in contours if cv2.contourArea(c) > 100]
    normalized_features = (object_features - object_features.mean(axis=0)) / object_features.std(axis=0)
    centroids, cluster_assignments = kmeans_clustering(normalized_features, k)
    print("Cluster assignments:", cluster_assignments)

    # Draw bounding boxes with different colors based on their cluster assignments
    image = cv2.imread(image_path)
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    for i, contour in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(contour)
        color = colors[cluster_assignments[i] % len(colors)]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    # Save the resulting image
    cv2.imwrite('clustered_objects.jpg', image)
    cv2.imshow('clustered_objects', image)
    cv2.waitKey(0)

if __name__ == "__main__":
    image_path = 'NIVETHANRAMESH_TP062192_PIC2.jpg'
    main(image_path, k=3)
