from rustem import EM

n_clusters = 32
img_path = "../images/img.jpg"
out_path = f"../images/output_k{n_clusters}.jpg"
epochs = 30

em = EM(img_path, n_clusters)
# fit the GMM
em.fit(epochs)
# Restructure the image and save it in the output path
em.restruct(out_path)
