from PIL import Image
import matplotlib.pyplot as plt

# Open an image file
img1 = Image.open('../fire_dataset/fire_images/fire.1.png')
img2 = Image.open('../fire_dataset/non_fire_images/non_fire.1.png')

# Create a subplot with 1 row and 2 columns
fig, axes = plt.subplots(nrows=1, ncols=2)

# Display the images
axes[0].imshow(img1)
axes[1].imshow(img2)

# Set titles
axes[0].set_title('Fire')
axes[1].set_title('Non Fire')

# Remove the axis
axes[0].axis('off')
axes[1].axis('off')

plt.show()
