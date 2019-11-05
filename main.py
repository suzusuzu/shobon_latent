import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from GPy.models.gplvm import GPLVM

def img_save(x, f_name):
    plt.figure(figsize=(12, 2))
    plt.scatter(x[:,0], x[:,1], c='black')
    plt.tick_params(labelbottom=False,
                    labelleft=False,
                    labelright=False,
                    labeltop=False)
    plt.tick_params(bottom=False,
                    left=False,
                    right=False,
                    top=False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.savefig(f_name)

if __name__ == "__main__":
    np.random.seed(3)

    image = np.asarray(Image.open('./data/shobon.png').convert('L'))

    # binarization
    a = np.where(image < 240)
    shobon = np.c_[a[1], np.flipud(a[0])]
    img_save(shobon, f_name='shobon.png')

    # high dimension
    dim = 100
    w = np.random.normal(size=2*dim).reshape(2, dim)
    high = shobon @ w
    high = high + np.random.normal(size=high.shape[0]*high.shape[1]).reshape(high.shape[0], high.shape[1])

    # latent
    gplvm = GPLVM(high, 2)
    latent = gplvm.X

    img_save(latent, f_name='latent.png')