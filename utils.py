import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

def img2tensor(img):
    img = img[:, :, :, None]
    img = img.transpose((3, 2, 0, 1))
    img = img.astype(np.float32)
    img = torch.from_numpy(img).cuda()
    return img

def imageE_preprocessing(path_msi,path_he,n_factor,m1,n1):
    img = np.loadtxt(path_msi)
    img = img.reshape(m1, n1, 3)
    img = cv2.resize(img, (n1 * n_factor, m1 * n_factor),interpolation = cv2.INTER_CUBIC)

    img2 = cv2.imread(path_he)
    img2 = cv2.resize(img2, (n1 * n_factor, m1 * n_factor),interpolation = cv2.INTER_CUBIC)
    img2 = img2 / 255

    img3 = np.loadtxt(path_msi)
    img3 = img3.reshape(m1, n1, 3)

    plt.subplot(1, 2, 1)
    plt.imshow(img3[:, :, [2, 1, 0]])
    plt.title('LR MSI')
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title('HR H&E')
    plt.show()

    imgg = img2tensor(img)
    imgg2 = img2tensor(img2)
    imgg3 = img2tensor(img3)

    return imgg,imgg2,imgg3

def imageI_preprocessing(path_msiI, path_he, path_msiE, n_factor, m1, n1):
    img = np.loadtxt(path_msiI)
    img = cv2.resize(img, (n1 * n_factor, m1 * n_factor),interpolation = cv2.INTER_CUBIC)
    img = img.reshape(m1 * n_factor, n1 * n_factor, 1)

    img2 = cv2.imread(path_he)
    img2 = cv2.resize(img2, (n1 * n_factor, m1 * n_factor),interpolation = cv2.INTER_CUBIC)
    img2 = img2 / 255

    img3 = np.loadtxt(path_msiI)
    img3 = img3.reshape(m1, n1, 1)

    plt.subplot(1, 2, 1)
    plt.imshow(img3.reshape(m1, n1))
    plt.title('LR MSI')
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title('HR H&E')
    plt.show()

    img4 = np.loadtxt(path_msiE)
    img4 = img4.reshape(m1, n1, 3)
    img4 = cv2.resize(img4, (n1 * n_factor, m1 * n_factor),interpolation = cv2.INTER_CUBIC)

    img5 = np.loadtxt(path_msiE)
    img5 = img5.reshape(m1, n1, 3)

    imgg = img2tensor(img)
    imgg2 = img2tensor(img2)
    imgg3 = img2tensor(img3)
    imgg4 = img2tensor(img4)
    imgg5 = img2tensor(img5)

    return imgg,imgg2,imgg3,imgg4,imgg5

def save_pic(output,j,output_file):
    to_imag = output.detach().cpu().numpy()[0]
    to_imag = to_imag.transpose((1, 2, 0))[:, :, 3:]
    cv2.imwrite(output_file + 'HR_fusion%d.jpg' % (j - 1000), to_imag * 255)

def save_txt(output, j, output_file):
    to_imag = output.detach().cpu().numpy()[0]
    to_imag = to_imag.transpose((1, 2, 0))
    np.savetxt(output_file +'HR_fusion%d.txt' % j, to_imag[:, :, 1])
    to_imag = output.detach().cpu().numpy()[0]
    to_imag = to_imag.transpose((1, 2, 0))
    np.savetxt(output_file + 'HR_fusion_2%d.txt' % j, to_imag[:, :, 0])
