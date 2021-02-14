## EM Image Segmentation

This is a simple implementation of the EM algorithm for performing image segmentation. The goal is to find the `k` components 
that contain the most information about the pixels of an image (a coloured image). 

This is done by first converting the 3d array of the image (one extra dimension for each `rgb` color) to a 2d representation. 
Then, on top of this new array, we apply the EM algorithm in order to approximate the parameters of the GMM that is defined
for the specified number of clusters/components.

There is a very high possibility that the algorithm is wrong since I actually implemented it 2 years ago without 
sufficient knowledge of the topic. The first implementation was in python (the jupyter notebook located under `pure-python`).
Recently, I started learning Rust and so I thought that this would be a good experiment to learn the language. 

I have used `PyO3` for binding my Rust code to a python module (named `rustem`) which can be invoked as `from rustem import EM`. 
My Rust code is definitely not the best but I am still learning.

## Example Output

Example script:
```python
from rustem import EM, save_new

epochs = 30
n_clusters = 32
img_path = "../images/img.jpg"

# Initialize EM
em = EM(img_path, n_clusters)
# Fit the GMM
em.fit(30)
# Save the new image
save_new(em, "path/to/new/image.jpg")
```
The output can bee seen in the image below:
![k=32 output](images/output_k32.png)

- Left: The initial image.
- Right: The new image after using only 32 colors.

## Performance


