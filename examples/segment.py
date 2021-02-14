from rustem import EM, save_new

epochs = 30
n_clusters = 32
img_path = "../images/img.jpg"

# Initialize EM
em = EM(img_path, n_clusters)
# Fit the GMM
em.fit(30)
# Save the new image
save_new(em, "/home/geoph/Documents/newimgds.jpg")
