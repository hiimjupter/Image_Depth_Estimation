import cv2
import numpy as np


def l1_distance(x, y):
    return abs(x - y)


def l2_distance(x, y):
    return (x - y) ** 2


left_img_path = 'tsukuba/left.png'
right_img_path = 'tsukuba/right.png'
disparity_range = 16

left = cv2.imread(left_img_path)
right = cv2.imread(right_img_path)

# Corrected cv2.imshow calls
cv2.imshow('Left Image', left)
cv2.imshow('Right Image', right)
cv2.waitKey(0)  # Wait for a key press to close the images
cv2.destroyAllWindows()


def pixel_wise_matching_l1(left_img, right_img, disparity_range, save_result=True):
    # Read left, right images then convert to grayscale
    left = cv2.imread(left_img, 0)
    right = cv2.imread(right_img, 0)

    left = left.astype(np.float32)
    right = right.astype(np.float32)

    height, width = left.shape[:2]

    # Create blank disparity map
    depth = np.zeros((height, width), np.uint8)
    scale = 16
    max_value = 255

    for y in range(height):
        for x in range(width):
            # Find j where cost has minimum value
            disparity = 0
            cost_min = max_value

            for j in range(disparity_range):
                cost = max_value if (x - j) < 0 \
                    else l1_distance(int(left[y, x]), int(right[y, x - j]))

                if cost < cost_min:
                    cost_min = cost
                    disparity = j

            # Set the depth at (y, x) equal to disparity scaled for visualization
            depth[y, x] = disparity * scale

    if save_result:
        print('Saving result...')
        # Save results
        cv2.imwrite(f'pixel_wise_l1.png', depth)
        cv2.imwrite(f'pixel_wise_l1_color.png',
                    cv2.applyColorMap(depth, cv2.COLORMAP_JET))

    print('Done.')

    return depth, cv2.applyColorMap(depth, cv2.COLORMAP_JET)


def pixel_wise_matching_l2(left_img, right_img, disparity_range, save_result=True):
    # Read left, right images then convert to grayscale
    left = cv2.imread(left_img, 0)
    right = cv2.imread(right_img, 0)

    left = left.astype(np.float32)
    right = right.astype(np.float32)

    height, width = left.shape[:2]

    # Create blank disparity map
    depth = np.zeros((height, width), np.uint8)
    scale = 16
    max_value = 255 ** 2

    for y in range(height):
        for x in range(width):
            # Find j where cost has minimum value
            disparity = 0
            cost_min = max_value

            for j in range(disparity_range):
                cost = max_value if (
                    x - j) < 0 else l2_distance(int(left[y, x]), int(right[y, x - j]))

                if cost < cost_min:
                    cost_min = cost
                    disparity = j

            # Set the depth at (y, x) equal to disparity scaled for visualization
            depth[y, x] = disparity * scale

    if save_result:
        print('Saving result...')
        # Save results
        cv2.imwrite(f'pixel_wise_l2.png', depth)
        cv2.imwrite(f'pixel_wise_l2_color.png',
                    cv2.applyColorMap(depth, cv2.COLORMAP_JET))

    print('Done.')

    return depth, cv2.applyColorMap(depth, cv2.COLORMAP_JET)


# Test case for L1 distance
depth, color = pixel_wise_matching_l1(
    left_img_path,
    right_img_path,
    disparity_range,
    save_result=True
)

# Uncomment to test L2 distance
# depth, color = pixel_wise_matching_l2(
#     left_img_path,
#     right_img_path,
#     disparity_range,
#     save_result=True
# )

cv2.imshow('Depth Map', depth)
cv2.imshow('Color Depth Map', color)
cv2.waitKey(0)
cv2.destroyAllWindows()
