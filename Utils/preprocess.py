import numpy as np
import cv2


def neg_image(img):
    img = np.array(img)
    neg_img = 255 - img
    return neg_img


def foreground(image, threshold=1000):
    column_sums = np.sum(image, axis=0)
    row_sums = np.sum(image, axis=1)

    left_index_col = np.argmax(column_sums > threshold)
    right_index_col = len(column_sums) - np.argmax(column_sums[::-1] > threshold) - 1

    left_index_row = np.argmax(row_sums > threshold)
    right_index_row = len(row_sums) - np.argmax(row_sums[::-1] > threshold) - 1

    return left_index_col, right_index_col, left_index_row, right_index_row


def cal_grav(img, img_size):
    moments = cv2.moments(img)
    if moments['m00'] == 0:
        cx, cy = img_size//2, img_size//2
        return cx, cy
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    return cx, cy


def get_patch(image, cx, cy, width=128):
    start_row = max(0, cy - width // 2)
    end_row = min(image.shape[0], cy + width // 2)
    start_col = max(0, cx - width // 2)
    end_col = min(image.shape[1], cx + width // 2)

    patch = image[start_row:end_row, start_col:end_col]
    if patch.shape[0] < width:
        padding_top = (width - patch.shape[0]) // 2
        padding_bottom = width - patch.shape[0] - padding_top
        patch = np.pad(patch, ((padding_top, padding_bottom), (0, 0)), "constant", constant_values=0)
    if patch.shape[1] < width:
        padding_left = (width - patch.shape[1]) // 2
        padding_right = width - patch.shape[1] - padding_left
        patch = np.pad(patch, ((0, 0), (padding_left, padding_right)), "constant", constant_values=0)

    return patch


def patching(img, img_size):
    img = neg_image(img)
    lc, rc, lr, rr = foreground(img)
    crop_img = img[lr:rr+1, lc:rc+1]
    cx, cy = cal_grav(crop_img, img_size)
    cx, cy = cx + lc, cy + lr
    patch0 = get_patch(img, cx, cy, img_size)
    patch1 = get_patch(img, cx-50, cy-50, img_size)
    patch2 = get_patch(img, cx-50, cy+50, img_size)
    patch3 = get_patch(img, cx+50, cy-50, img_size)
    patch4 = get_patch(img, cx+50, cy+50, img_size)
    patches = [patch0, patch1, patch2, patch3, patch4]
    patches = [patch for patch in patches if (np.sum(patch > 0) / patch.shape[0]**2) >= 0.6]
    if len(patches) == 0:
        patches.append(patch0)
    # for patch in patches:
    #     if (np.sum(patch > 0) / patch.shape[0]**2) < 0.6:
    #         patches.remove(patch.all())
    return patches


def augmentation(img, label):
    aug_img = []
    rot_img = []
    for degree in range(1, 4):
        rotate_img = np.rot90(img, k=degree)
        rot_img.append(rotate_img)

    for image in rot_img:
        aug_img.append(image)
        hor_flip = np.flip(image, axis=1)
        aug_img.append(hor_flip)
        ver_flip = np.flip(image, axis=0)
        aug_img.append(ver_flip)
    hor_flip = np.flip(img, axis=1)
    ver_flip = np.flip(img, axis=0)
    aug_img.append(hor_flip)
    aug_img.append(ver_flip)

    aug_label = [label for _ in range(len(aug_img))]
    return aug_img, aug_label
