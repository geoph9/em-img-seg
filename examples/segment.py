from rustem import EM, save_new

n_clusters = 32
img_path = "../images/img.jpg"
out_path = f"../images/output_k{n_clusters}.jpg"
epochs = 30

# Initialize EM
em = EM(img_path, n_clusters)
# Fit the GMM
em.fit(epochs)
# Save the new image
save_new(em, out_path)
