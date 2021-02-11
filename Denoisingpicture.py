
import numpy as np
import matplotlib.pyplot as plt

# Hi guys , I'm Mohammad Javad Rajabi, so let's dive in 

# reading image
# mypath = "noisy.jpg"
mypath = "your path"
myimage = plt.imread(mypath)

# forming SVD decomposition
myimage = myimage / 256
myimageT = np.transpose(myimage )
u , s ,vt = np.linalg.svd(myimageT)

Sigma = np.zeros((3, u.shape[2], vt.shape[1]))
for j in range(3):
    np.fill_diagonal(Sigma[j, :, :], s[j, :])

reconstructed =  u @ Sigma @ vt


# denoising
k = 15
approx_img = u @ Sigma[..., :k] @ vt[..., :k, :]
# approx_img = np.array(approx_img,np.uint8)
approx_img = np.transpose(approx_img)



# showing 
plt.imshow(approx_img)
plt.show()


