from filters import DoGFilter, GaborFilter
from Encoders import TimeToFirstSpikeEncoding, Poisson
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# apply the filter to an image
def apply_filter(image, filter):
    # convert image to torch tensor
    image_tensor = torch.tensor(image, dtype=torch.float32)
    
    # apply the filter using convolution
    filter_size = filter.shape[0]
    padding = filter_size // 2
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0) 
    filter = filter.unsqueeze(0).unsqueeze(0) 
    filtered_image = torch.nn.functional.conv2d(image_tensor, filter, padding=padding)
    
    return filtered_image.squeeze().numpy()


# read an image and convert it to torch tensor
def image_to_vec(img1, size=(10, 10)):
    img = cv2.resize(img1, (size[0], size[1]))
    img = torch.from_numpy(img)

    return img


def show_image(img):
    plt.imshow(img, cmap='gray')  
    plt.show()


def raster_plot(spikes, fsize=(5,5)):
    
    plt.figure(figsize=fsize)
    plt.xlim(0, len(spikes))
    s_spikes = torch.nonzero(spikes)
    plt.scatter(s_spikes[:,0], s_spikes[:,1], s=2, c='darkviolet')
            
    plt.xlabel("Time")
    plt.ylabel("Neurons")
    plt.show()


def visualize_spikes(spikes, title):
    plt.figure(figsize=(10, 5))
    plt.imshow(spikes.sum(dim=0).cpu().numpy(), cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

image1 = cv2.imread('images/bird.tif', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('images/camera.tif', cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread('images/circles.tif', cv2.IMREAD_GRAYSCALE)
image4 = cv2.imread('images/lena1.tif', cv2.IMREAD_GRAYSCALE)
images = [image1, image2, image3, image4]

for i in range(1, 5):
    plt.subplot(2, 2, i)
    plt.title(f'Image {i}')
    plt.imshow(images[i-1], cmap='gray')
    plt.axis('off')


on_center_dog_filter = DoGFilter(size=15, sigma_1=1.0, sigma_2=5.0, dtype=torch.float32)
off_center_dog_filter = DoGFilter(size=5, sigma_1=3.0, sigma_2=1.0, dtype=torch.float32)


on_filtered_images = []
for i in range(1, 5):
    on_filtered_images.append(apply_filter(images[i-1], on_center_dog_filter))

off_filtered_images = []
for i in range(1, 5):
    off_filtered_images.append(apply_filter(images[i-1], off_center_dog_filter))


plt.figure(figsize=(10, 5))

for i in range(1, 5):
    plt.subplot(2, 2, i)
    plt.title(f'Filtered Image {i}')
    plt.imshow(on_filtered_images[i-1], cmap='gray')
    plt.axis('off')

plt.show()

plt.figure(figsize=(10, 5))

for i in range(1, 5):
    plt.subplot(2, 2, i)
    plt.title(f'Filtered Image {i}')
    plt.imshow(off_filtered_images[i-1], cmap='gray')
    plt.axis('off')

plt.show()


gabor_filter = GaborFilter(size=20, labda=10.0, theta=np.pi/4, sigma=5.0, gamma=1.0, phi=0, dtype=torch.float32)


filtered_images = []
for i in range(1, 5):
    filtered_images.append(apply_filter(images[i-1], gabor_filter))


plt.figure(figsize=(10, 5))

for i in range(1, 5):
    plt.subplot(2, 2, i)
    plt.title(f'Filtered Image {i}')
    plt.imshow(filtered_images[i-1], cmap='gray')
    plt.axis('off')

plt.show()


ttfs_on_dogs, ttfs_off_dogs, ttfs_gabor = [], [], []
for i in range(4):
    img = image_to_vec(on_filtered_images[i], (50, 50))
    ttfs = TimeToFirstSpikeEncoding(img, 1000)
    ttfs_on_dogs.append(ttfs.encode())


    img = image_to_vec(off_filtered_images[i], (50, 50))
    ttfs = TimeToFirstSpikeEncoding(img, 1000)
    ttfs_off_dogs.append(ttfs.encode())

    img = image_to_vec(filtered_images[i], (50, 50))
    ttfs = TimeToFirstSpikeEncoding(img, 1000)
    ttfs_gabor.append(ttfs.encode())

for pl in ttfs_on_dogs:
    raster_plot(pl)

for pl in ttfs_off_dogs:
    raster_plot(pl)

for pl in ttfs_gabor:
    raster_plot(pl)


normalized_on_dog, normalized_off_dog, normalized_gabor = [], [], []
for i in range(4):
    filtered_img_norm = (on_filtered_images[i] - on_filtered_images[i].min()) / (on_filtered_images[i].max() - on_filtered_images[i].min())
    # convert to torch tensors
    filtered_img_tensor = torch.tensor(filtered_img_norm, dtype=torch.float32)
    normalized_on_dog.append(filtered_img_tensor)

    filtered_img_norm = (off_filtered_images[i] - off_filtered_images[i].min()) / (off_filtered_images[i].max() - off_filtered_images[i].min())
    # convert to torch tensors
    filtered_img_tensor = torch.tensor(filtered_img_norm, dtype=torch.float32)
    normalized_off_dog.append(filtered_img_tensor)

    filtered_img_norm = (filtered_images[i] - filtered_images[i].min()) / (filtered_images[i].max() - filtered_images[i].min())
    # convert to torch tensors
    filtered_img_tensor = torch.tensor(filtered_img_norm, dtype=torch.float32)
    normalized_gabor.append(filtered_img_tensor)


# create the Poisson encoder
poisson_encoder = Poisson(time_window=50, ratio=5.0)
# generate the spike trains
on_encoding, off_encoding, g_encoding = [], [], []
for i in range(4):
    spikes_image = poisson_encoder(normalized_on_dog[i])
    on_encoding.append(spikes_image)

    spikes_image = poisson_encoder(normalized_off_dog[i])
    off_encoding.append(spikes_image)

    spikes_image = poisson_encoder(normalized_gabor[i])
    g_encoding.append(spikes_image)


# visualize the spike trains
for i in range(4):
    visualize_spikes(on_encoding[i], f'Spike Train for Filtered Image {i+1} with on-centered DoG')
    
for i in range(4):
    visualize_spikes(off_encoding[i], f'Spike Train for Filtered Image {i+1} with off-centered DoG')

for i in range(4):
    visualize_spikes(g_encoding[i], f'Spike Train for Filtered Image {i+1} with Gabor filter')


# Load the image
image_rgb1 = cv2.imread('images/lena3.tif')
image_rgb1 = cv2.cvtColor(image_rgb1, cv2.COLOR_BGR2RGB)
image_rgb2 = cv2.imread('images/monarch.tif')
image_rgb2 = cv2.cvtColor(image_rgb2, cv2.COLOR_BGR2RGB)
image_rgb3 = cv2.imread('images/peppers3.tif')
image_rgb3 = cv2.cvtColor(image_rgb3, cv2.COLOR_BGR2RGB)
image_rgb4 = cv2.imread('images/tulips.tif')
image_rgb4 = cv2.cvtColor(image_rgb4, cv2.COLOR_BGR2RGB)

rgb_images = [image_rgb1, image_rgb2, image_rgb3, image_rgb4]

for i in range(1, 5):
    plt.subplot(2, 2, i)
    plt.title(f'Image {i}')
    plt.imshow(rgb_images[i-1])
    plt.axis('off')


image_rs, image_gs, image_bs = [], [], []
for i in range(4):
    image_r = rgb_images[i][:, :, 0]
    image_g = rgb_images[i][:, :, 1]
    image_b = rgb_images[i][:, :, 2]

    image_rs.append(image_r)
    image_gs.append(image_g)
    image_bs.append(image_b)


on_center_dog_filter = DoGFilter(size=20, sigma_1=1.0, sigma_2=8.0, dtype=torch.float32)
off_center_dog_filter = DoGFilter(size=5, sigma_1=3.0, sigma_2=1.0, dtype=torch.float32)

on_filtered_rgbs, off_filtered_rgbs = [], []
for i in range(4):
    # on-centered
    # apply the DoG filter to each channel
    filtered_r = apply_filter(image_rs[i], on_center_dog_filter)
    filtered_g = apply_filter(image_gs[i], on_center_dog_filter)
    filtered_b = apply_filter(image_bs[i], on_center_dog_filter)

    # normalize the filtered channels
    filtered_r_norm = (filtered_r - filtered_r.min()) / (filtered_r.max() - filtered_r.min())
    filtered_g_norm = (filtered_g - filtered_g.min()) / (filtered_g.max() - filtered_g.min())
    filtered_b_norm = (filtered_b - filtered_b.min()) / (filtered_b.max() - filtered_b.min())

    # combine the filtered channels back into an RGB image
    filtered_rgb = np.stack((filtered_r_norm, filtered_g_norm, filtered_b_norm), axis=-1)
    on_filtered_rgbs.append(filtered_rgb)

    # off-centered
    # apply the DoG filter to each channel
    filtered_r = apply_filter(image_rs[i], off_center_dog_filter)
    filtered_g = apply_filter(image_gs[i], off_center_dog_filter)
    filtered_b = apply_filter(image_bs[i], off_center_dog_filter)

    # normalize the filtered channels
    filtered_r_norm = (filtered_r - filtered_r.min()) / (filtered_r.max() - filtered_r.min())
    filtered_g_norm = (filtered_g - filtered_g.min()) / (filtered_g.max() - filtered_g.min())
    filtered_b_norm = (filtered_b - filtered_b.min()) / (filtered_b.max() - filtered_b.min())

    # combine the filtered channels back into an RGB image
    filtered_rgb = np.stack((filtered_r_norm, filtered_g_norm, filtered_b_norm), axis=-1)
    off_filtered_rgbs.append(filtered_rgb)


    plt.figure(figsize=(10, 5))

for i in range(1, 5):
    plt.subplot(2, 2, i)
    plt.title(f'on-centered Dog Filtered Image {i}')
    plt.imshow(on_filtered_rgbs[i-1])
    plt.axis('off')

plt.show()

plt.figure(figsize=(10, 5))

for i in range(1, 5):
    plt.subplot(2, 2, i)
    plt.title(f'off-centered DoG Filtered Image {i}')
    plt.imshow(off_filtered_rgbs[i-1])
    plt.axis('off')

plt.show()