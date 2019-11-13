import errno, os, sys
import glob
import numpy as np
import cv2
import math

def pad_to_image(img, target_size):
    pad_amount = [0, 0, 0, 0]
    (h, w) = img.shape[0:2]
    extra_h = target_size[1] - h
    extra_w = target_size[0] - w

    e_h1 = int(math.ceil(extra_h/2.0))
    e_h2 = extra_h - e_h1

    e_w1 = int(math.ceil(extra_w/2.0))
    e_w2 = extra_w - e_w1

    pad_amount = ((e_h1, e_h2), (e_w1, e_w2))

    if len(img.shape) == 3:
        pad_amount = (pad_amount[0], pad_amount[1], (0, 0))

    out_img = np.pad(img, pad_amount, 'constant')

    return (out_img, pad_amount)

def get_all_subdirs(dir_name):
    lst = os.listdir(dir_name)
    ret = [os.path.join(dir_name, e) for e in lst if os.path.isdir(os.path.join(dir_name, e))]
    return ret


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def getImageFilesInFold(dir_path, extensions=['.jpg', '.jpeg', '.png', '.tif', '.JPG', '.PNG'], recursive=False):
    im_list = []
    extensions = tuple(extensions)
    if (sys.version_info >= (3, 0)):
        # Python 3.x
        for filename in glob.iglob(os.path.join(dir_path, '**'), recursive=recursive):
            if filename.endswith(extensions):
                im_list.append(filename)
    else:
        # Python 2.x
        for r, d, f in os.walk(dir_path):
            for image in f:
                if image.endswith(extensions) and not image.startswith('.'):
                    image_path = os.path.join(r, image)
                    # print (image_path)
                    im_list.append(image_path)

    im_list = sorted(im_list)
    return im_list



def find_enclosing_rect(mask, pix_border=1):
    vsum = np.sum(mask, axis=0)  # Sums vertically. Final length is equal to width of image
    hsum = np.sum(mask, axis=1)  # Sums horizontally along a row. Final length is equal to heigh of image

    y1 = hsum.size - 1
    y2 = 0
    for (i, val) in enumerate(hsum):
        if val != 0:
            y1 = i
            break

    for (i, val) in enumerate(reversed(hsum)):
        if val != 0:
            y2 = hsum.size - i
            break

    x1 = vsum.size - 1
    x2 = 0
    for (i, val) in enumerate(vsum):
        if val != 0:
            x1 = i
            break

    for (i, val) in enumerate(reversed(vsum)):
        if val != 0:
            x2 = vsum.size - i
            break

    x1 = max(0, x1 - pix_border)
    y1 = max(0, y1 - pix_border)
    x2 = min(x2 + pix_border, vsum.size - 1)
    y2 = min(y2 + pix_border, hsum.size - 1)

    rect = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
    if rect[2] <= 0 or rect[3] <=0:
        rect[2] = 0
        rect[3] = 0

    return tuple(rect)

def rotate_bound(image, angle):
    # Reference: https://github.com/jrosebr1/imutils/blob/master/imutils/convenience.py
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))