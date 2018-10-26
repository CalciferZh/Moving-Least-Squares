import numpy as np
import pickle
from imageio import imread, imsave
import cv2
import random

from img_utils import (mls_affine_deformation, mls_affine_deformation_inv,
                       mls_similarity_deformation, mls_similarity_deformation_inv,
                       mls_rigid_deformation, mls_rigid_deformation_inv)

from calciferzh_utils import *

required_keypoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

def batch_wrapper():
  # mls_rigid_deformation_inv
  # mls_affine_deformation_inv
  func = mls_similarity_deformation_inv
  # arr_to_tuple = lambda arr: tuple(int(round(i)) for i in arr)
  data_pack = pkl_load('./data_pack.pkl')
  random.seed(9608)
  random.shuffle(data_pack)
  for pack in data_pack:
    name, keypoints, new_keypoints = pack
    keypoints = np.round(keypoints)[required_keypoints]
    new_keypoints = np.round(new_keypoints)[required_keypoints]
    img = imread('./masks/%s.png' % name)
    # for p in keypoints:
    #   cv2.circle(
    #     img,
    #     arr_to_tuple(p),
    #     5,
    #     (255, 255, 255),
    #     -1
    #   )
    deformed_img = func(img, keypoints, new_keypoints, alpha=1, density=1)
    # for p in new_keypoints:
    #   cv2.circle(
    #     deformed_img,
    #     arr_to_tuple(p),
    #     5,
    #     (255, 255, 255),
    #     -1
    #   )
    # print(name)
    # imshow_onerow([img, deformed_img])

    imsave('./result/%s.png' % name, deformed_img)

    # dpi = 144 # my monitor
    # fig = plt.figure(figsize=(img.shape[1]*2*1.2/dpi, img.shape[0]*1.2/dpi), dpi=dpi)
    # fig.add_subplot(1, 2, 1)
    # plt.imshow(img)
    # fig.add_subplot(1, 2, 2)
    # plt.imshow(deformed_img)
    # plt.savefig('./deformation/%s.png' % name, dpi=dpi)
    # plt.close(fig)


if __name__ == '__main__':
  batch_wrapper()
