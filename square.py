import cv2


def squarize(img):
    # Get the height and width of the image
    h, w = img.shape[:2]

    # Calculate the size of the square image
    size = min(w, h)

    # Calculate the coordinates of the top-left corner of the crop region
    x = (w - size) // 2
    y = (h - size) // 2

    # Crop the image to the square region
    cropped_img = img[y:y+size, x:x+size]

    # Resize the image to the desired dimensions
    new_size = 512  # desired size of the square image
    resized_img = cv2.resize(cropped_img, (new_size, new_size))
    return resized_img
