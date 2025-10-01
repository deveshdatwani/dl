from utils.dataloader import BSDS500Dataset
from skimage.segmentation import find_boundaries
from matplotlib import pyplot as plt


dataset = BSDS500Dataset('/home/deveshdatwani/dl/data/bsds500', split='train')
img, gts = dataset[150]
fig, axs = plt.subplots(1, 2, figsize=(128, 128)) 
axs[0].imshow(gts, cmap='gray')
axs[1].imshow(img)
plt.tight_layout()
plt.show()