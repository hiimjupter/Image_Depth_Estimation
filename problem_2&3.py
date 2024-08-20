import cv2
import numpy as np


def l1_distance(x, y):
    return abs(x - y)


def l2_distance(x, y):
    return (x - y) ** 2


def window_based_matching_l1(left_img, right_img, disparity_range, kernel_size=5, save_result=True):
    # Read left, right images then convert to grayscale
    left = cv2.imread(left_img, 0)
    right = cv2.imread(right_img, 0)

    # Check if images were loaded correctly
    if left is None:
        print(f"Error: Could not load image at {left_img}")
        return
    if right is None:
        print(f"Error: Could not load image at {right_img}")
        return

    left = left.astype(np.float32)
    right = right.astype(np.float32)

    height, width = left.shape[:2]

    # Create blank disparity map
    depth = np.zeros((height, width), np.uint8)

    kernel_half = int((kernel_size - 1) / 2)
    scale = 3
    max_value = 255 * kernel_size * kernel_size

    for y in range(kernel_half, height - kernel_half):
        for x in range(kernel_half, width - kernel_half):
            # Find j where cost has minimum value
            disparity = 0
            cost_min = max_value

            for j in range(disparity_range):
                total = 0
                value = 0

                for v in range(-kernel_half, kernel_half + 1):
                    for u in range(-kernel_half, kernel_half + 1):
                        value = max_value
                        if (x + u - j) >= 0:
                            value = l1_distance(
                                int(left[y + v, x + u]), int(right[y + v, (x + u) - j]))
                        total += value

                if total < cost_min:
                    cost_min = total
                    disparity = j

            # Let depth at (y, x) = j (disparity)
            # Multiply by a scale factor for visualization purpose
            depth[y, x] = disparity * scale

    if save_result:
        print('Saving result...')
        # Save results
        cv2.imwrite('window_based_l1.png', depth)
        cv2.imwrite('window_based_l1_color.png',
                    cv2.applyColorMap(depth, cv2.COLORMAP_JET))

    print('Done.')

    return depth, cv2.applyColorMap(depth, cv2.COLORMAP_JET)


def window_based_matching_l2(left_img, right_img, disparity_range, kernel_size=5, save_result=True):
    # Read left, right images then convert to grayscale
    left = cv2.imread(left_img, 0)
    right = cv2.imread(right_img, 0)

    # Check if images were loaded correctly
    if left is None:
        print(f"Error: Could not load image at {left_img}")
        return
    if right is None:
        print(f"Error: Could not load image at {right_img}")
        return

    left = left.astype(np.float32)
    right = right.astype(np.float32)

    height, width = left.shape[:2]

    # Create blank disparity map
    depth = np.zeros((height, width), np.uint8)

    kernel_half = int((kernel_size - 1) / 2)
    scale = 3
    max_value = 255 ** 2 * kernel_size * kernel_size

    for y in range(kernel_half, height - kernel_half):
        for x in range(kernel_half, width - kernel_half):

            # Find j where cost has minimum value
            disparity = 0
            cost_min = max_value

            for j in range(disparity_range):
                total = 0
                value = 0

                for v in range(-kernel_half, kernel_half + 1):
                    for u in range(-kernel_half, kernel_half + 1):
                        value = max_value
                        if (x + u - j) >= 0:
                            value = l2_distance(
                                int(left[y + v, x + u]), int(right[y + v, (x + u) - j]))
                        total += value

                if total < cost_min:
                    cost_min = total
                    disparity = j

            # Let depth at (y, x) = j (disparity)
            # Multiply by a scale factor for visualization purpose
            depth[y, x] = disparity * scale

    if save_result:
        print('Saving result...')
        # Save results
        cv2.imwrite('window_based_l2.png', depth)
        cv2.imwrite('window_based_l2_color.png',
                    cv2.applyColorMap(depth, cv2.COLORMAP_JET))

    print('Done.')

    return depth, cv2.applyColorMap(depth, cv2.COLORMAP_JET)


left_img_path = 'Aloe/Aloe_left_1.png'
# Uncomment for Problem 2
# right_img_path = 'Aloe/Aloe_right_1.png'
# Problem 3
# Remember to comment this when uncomment problem 2
right_img_path = 'Aloe/Aloe_right_2.png'
disparity_range = 64
kernel_size = 3

left = cv2.imread(left_img_path)
right = cv2.imread(right_img_path)

# Check if images were loaded correctly
if left is None:
    print(f"Error: Could not load image at {left_img_path}")
else:
    cv2.imshow('Left Image', left)

if right is None:
    print(f"Error: Could not load image at {right_img_path}")
else:
    cv2.imshow('Right Image', right)

if left is not None and right is not None:
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # L1 Result
    depth, color_depth = window_based_matching_l1(
        left_img_path,
        right_img_path,
        disparity_range,
        kernel_size=kernel_size,
        save_result=True
    )

    cv2.imshow('L1 Depth Map', depth)
    cv2.imshow('L1 Color Depth Map', color_depth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # L2 Result
    depth, color_depth = window_based_matching_l2(
        left_img_path,
        right_img_path,
        disparity_range,
        kernel_size=kernel_size,
        save_result=True
    )

    cv2.imshow('L2 Depth Map', depth)
    cv2.imshow('L2 Color Depth Map', color_depth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
