from utils.dataloader import BSDS500Dataset
from skimage.segmentation import find_boundaries
from matplotlib import pyplot as plt


dataset = BSDS500Dataset('/home/deveshdatwani/dl/data/bsds500', split='train')
img, gts = dataset[0]
seg = gts[0]
edges = find_boundaries(seg, mode='outer')
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.title("Edge map from segmentation")
plt.show()