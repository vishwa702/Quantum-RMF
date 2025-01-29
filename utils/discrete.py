import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssi
import tensorflow as tf
import cv2

# D-Wave
from dwave.samplers import SimulatedAnnealingSampler
import dimod
from dimod import CQM, Binary, Integer, quicksum, cqm_to_bqm, BQM
from dwave.system import DWaveSampler, LeapHybridSampler, LeapHybridCQMSampler, LeapHybridNLSampler
from dwave.system.composites import EmbeddingComposite


# Import environment variables
import dotenv
dotenv.load_dotenv()
solver_token = os.getenv('SOLVER_TOKEN')


BASE_DIR = ''
RESULTS_DIR = BASE_DIR + 'results/'
RUNS_DIR = BASE_DIR + 'runs/'

def save_image(image_array, path):
    """
    Saves a numpy array as an image using OpenCV, handling pixel values in different ranges.

    Parameters:
    - image_array: 2D numpy array representing the image
    - path: str - path to save the image
    """

    # Normalize based on the maximum value
    max_value = np.max(image_array)
    if max_value <= 32:
        image_array = np.round((image_array / max_value) * 255).astype(np.uint8)
    elif max_value <= 1:
        image_array = np.round(image_array * 255).astype(np.uint8)

    image = image_array.astype(np.uint8)
    cv2.imwrite(path, image)

# ----------------- Preprocessing --------------------------

def add_gaussian_noise(images, noise_factor=0.1, precision=4):
    """
    Adds Gaussian noise to a set of images, and scales the pixel values to a discrete range [0, precision-1].

    Parameters:
    - images: numpy array of shape (num_images, height, width) containing the original images
    - noise_factor: float, standard deviation of the Gaussian noise to be added (default: 0.1)
    - precision: int, number of discrete levels for pixel intensities (default: 4)

    Returns:
    - noisy_images: numpy array of shape (num_images, height, width) containing noisy images
    """
    noisy_images = images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape)

    # Clip and scale to the range [0, precision-1]
    noisy_images = np.clip(noisy_images, 0, 1)
    noisy_images = np.round(noisy_images * (precision - 1)).astype(int)

    return noisy_images

def add_sp_noise(images, noise_factor=0.1, precision=4):
    """
    Adds salt-and-pepper noise to a set of images, and scales the pixel values to a discrete range [0, precision-1].

    Parameters:
    - images: numpy array of shape (num_images, height, width) containing the original images
    - noise_factor: float, proportion of image pixels to be replaced with noise (default: 0.1)
    - precision: int, number of discrete levels for pixel intensities (default: 4)

    Returns:
    - noisy_images: numpy array of shape (num_images, height, width) containing noisy images
    """
    noisy_images = images.copy()
    num_salt = np.ceil(noise_factor * images.size * 0.5)
    num_pepper = np.ceil(noise_factor * images.size * 0.5)

    # Salt noise
    salt_coords = [np.random.randint(0, i - 1, int(num_salt)) for i in images.shape]
    noisy_images[tuple(salt_coords)] = precision - 1  # maximum value for salt

    # Pepper noise
    pepper_coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in images.shape]
    noisy_images[tuple(pepper_coords)] = 0  # minimum value for pepper

    return noisy_images

# ----------------- Visualization --------------------------

def show_image(image, title=None):
  if image.ndim == 1:
    dim = int(np.sqrt(image.shape[0]))
    image = image.reshape((dim, dim))

  plt.imshow(image, cmap='gray')
  plt.title(title)
  plt.show()

def show_comparison(noise_free_images, noisy_images, reconstructed_images):
    """
    Plots Noise-Free, Noisy, and Reconstructed images in a grid.

    Parameters:
    - noise_free_images: List of noise-free images
    - noisy_images: List of noisy images
    - reconstructed_images: List of reconstructed images
    """

    if not isinstance(noise_free_images, list):
        noise_free_images = [noise_free_images]
    if not isinstance(noisy_images, list):
        noisy_images = [noisy_images]
    if not isinstance(reconstructed_images, list):
        reconstructed_images = [reconstructed_images]

    # Determine the number of images
    num_images = len(noise_free_images)

    # Create a figure with 3 columns and num_images rows
    fig, axes = plt.subplots(num_images, 3, figsize=(9, 3 * num_images + 1))

    # If there's only one set of images, ensure axes is 2D array for consistency
    if num_images == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_images):
        # Plot Noise-Free image
        axes[i, 0].imshow(noise_free_images[i], cmap='gray')
        if i == 0:
            axes[i, 0].set_title('Noise-Free')
        axes[i, 0].axis('off')  # Hide axes

        # Plot Noisy image
        axes[i, 1].imshow(noisy_images[i], cmap='gray')
        if i == 0:

            axes[i, 1].set_title('Noisy')
        axes[i, 1].axis('off')  # Hide axes

        # Plot Reconstructed image
        axes[i, 2].imshow(reconstructed_images[i], cmap='gray')
        if i == 0:
            axes[i, 2].set_title('Reconstructed')
        axes[i, 2].axis('off')  # Hide axes

    # Adjust layout
    # fig.suptitle(f'Results on noise factor {noise_factor}')
    # plt.tight_layout()
    plt.show()



def show_recons(recons):
  """
    Input:
      recons: np.array - An array of flattened images

    Displays reconstructed images.
  """

  num_images = recons.shape[0]

  if num_images == 0:
    print('No reconstructions available')
    return

  if num_images == 1:
    show_image(recons[0], title='Solution')
    return

  # Calculate the number of rows and columns for subplots
  num_rows = int(np.ceil(np.sqrt(num_images)))
  num_cols = int(np.ceil(num_images / num_rows))

  # Create subplots
  fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 12))

  # Iterate through the subplots and plot the images
  for i, ax in enumerate(axes.flat):
      if i < num_images:
          img = recons[i]
          ax.imshow(img.reshape((12, 12)), cmap='gray')
          ax.set_title(f'Solution {i+1}')
      else:
          ax.axis('off')  # Turn off empty subplots if there are extras

  # Adjust layout for better visualization
  plt.tight_layout()
  plt.show()



# ----------------- Evaluation --------------------------


def evaluate_ssi(x_train, x_train_noisy):
  """
    Input:
      x_train: np.array - Clean training images
      x_train_noisy: np.array - Noisy training images or reconstructed images

    Output:
      ssi: float - Structural Similarity Index
  """

  if x_train.shape != x_train_noisy.shape:
    if x_train.ndim == 3:
      x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]**2)).T
    elif x_train.ndim == 2 and x_train.shape[::-1] == x_train_noisy.shape:
      x_train = x_train.T
    else:
      raise ValueError(f'x_train and x_train_noisy must have the same shape. Found {x_train.shape} and {x_train_noisy.shape}')

  if x_train.dtype != x_train_noisy.dtype:
    x_train = x_train.astype(x_train_noisy.dtype)

  ssis = np.zeros(x_train.shape[0])
  for i,  (image, noisy_image) in enumerate(zip(x_train, x_train_noisy)):
    ssis[i] = ssi(image, noisy_image)

  return np.mean(ssis)

def get_l1_loss(X, X_noisy, show_diff=False):
  l1_loss = np.sum(np.abs(X-X_noisy))
  l1_loss /= np.max(X)
  if show_diff:
    plt.imshow(abs(X-X_noisy), cmap='gray')
    plt.figtext(0.5, 0, f'L1 Loss = {l1_loss}', wrap=True, horizontalalignment='center', fontsize=12)

  return l1_loss

def get_l2_loss(image1, image2):
    """
    Compute the L2 loss (mean squared error) between two images.

    Parameters:
    image1 (numpy.ndarray): The first image.
    image2 (numpy.ndarray): The second image.

    Returns:
    float: The L2 loss between the two images.
    """
    assert image1.shape == image2.shape, "Images must have the same shape"

    squared_diff = np.square(image1 - image2)
    sum_squared_diff = np.sum(squared_diff)
    mse = sum_squared_diff / image1.size

    return mse

def get_psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')

    # Define the maximum possible pixel value of the image
    pixel_max = max(np.max(image1), np.max(image2))

    # Calculate PSNR
    psnr = 20 * np.log10(pixel_max / np.sqrt(mse))
    return psnr


def show_energy(X, X_noisy):

  energy_neighbors = X.flatten() @ initialize_Q(X.shape) @ X.flatten()
  energy = X.flatten() @ np.diag(-2 * X_noisy.flatten() + 1) @ X.flatten()

  print(f'Energy Neighbors: {energy_neighbors}')
  print(f'Energy Retention: {energy - energy_neighbors}')
  print(f'Energy Total: {energy:.4f}')

# ----------------- Computation --------------------------

def reshape_data(images,  image_shape = (32, 32)):
  """
  Assuming input of shape (n_images, height, width)
  """

  # Convert to TensorFlow tensor and add a channel dimension
  images = tf.expand_dims(images, axis=-1)

  # Resize the images
  images_resized = tf.image.resize(images, image_shape)

  # Convert back to NumPy array if needed
  images_resized = images_resized.numpy()

  # Remove the single channel dimension to match the desired shape
  images_resized = np.squeeze(images_resized, axis=-1)

  # # Verify the shapes
  # print(f'Original shape: {images.shape}')
  # print(f'Resized shape: {images_resized.shape}')
  return images_resized


def sample_qubo(Q, sampler, label=''):
  """
    Input:
      Q: np.array - QUBO matrix or BQM object
      sampler: dwave.sampler - Sampler to be used for annealing
      show_solution: bool - to show reconstructed image

    Output:
      sampleset: dimod.SampleSet - sampleset of annealing solutions
  """

  if isinstance(Q, BQM):
     sampler.sample(Q).aggregate()
  else:
    sampleset = sampler.sample_qubo(Q, label=label).aggregate()

  return sampleset


def get_neighbor(i, j, n=28, m=28, order=1):
  """
  Return a list of neighbors of (i, j) in a 2D grid of size (n, m).
  The function supports different orders of neighborhood that include an increasing number of neighboring cells.
  The function also checks if the input indices are valid and within the bounds of the grid.

  Input:
    i: int - row index
    j: int - column index
    n: int - number of rows
    m: int - number of columns
    order: int - order of neighborhood

  Output:
    neighbors: list of tuples - neighbors of (i, j)
  """

  assert i >= 0 and i < n, "i is invalid"
  assert j >= 0 and j < m, "j is invalid"

  neighbors = []

  if order >= 1:
    if i - 1 >= 0:
      neighbors.append((i - 1, j))
    if i + 1 < n:
      neighbors.append((i + 1, j))
    if j - 1 >= 0:
      neighbors.append((i, j - 1))
    if j + 1 < m:
      neighbors.append((i, j + 1))

  if order >= 2:
    if i - 1 >= 0 and j - 1 >= 0:
      neighbors.append((i - 1, j - 1))
    if i - 1 >= 0 and j + 1 < m:
      neighbors.append((i - 1, j + 1))
    if i + 1 < n and j - 1 >= 0:
      neighbors.append((i + 1, j - 1))
    if i + 1 < n and j + 1 < m:
      neighbors.append((i + 1, j + 1))

  if order >= 3:
    if i - 2 >= 0:
      neighbors.append((i - 2, j))
    if i + 2 < n:
      neighbors.append((i + 2, j))
    if j - 2 >= 0:
      neighbors.append((i, j - 2))
    if j + 2 < m:
      neighbors.append((i, j + 2))

  if order >= 4:
    if i - 2 >= 0 and j - 1 >= 0:
      neighbors.append((i - 2, j - 1))
    if i - 2 >= 0 and j + 1 < m:
      neighbors
    if i - 1 >= 0 and j - 2 >= 0:
      neighbors.append((i - 1, j - 2))
    if i - 1 >= 0 and j + 2 < m:
      neighbors.append((i - 1, j + 2))

    if i + 2 < n and j - 1 >= 0:
      neighbors.append((i + 2, j - 1))
    if i + 2 < n and j + 1 < m:
      neighbors.append((i + 2, j + 1))
    if i + 1 < n and j - 2 >= 0:
      neighbors.append((i + 1, j - 2))
    if i + 1 < n and j + 2 < m:
      neighbors.append((i + 1, j + 2))

  if order == 5:
      if i - 2 >= 0 and j - 2 >= 0:
        neighbors.append((i - 2, j - 2))
      if i - 2 >= 0 and j + 2 < m:
        neighbors.append((i - 2, j + 2))
      if i + 2 < n and j - 2 >= 0:
        neighbors.append((i + 2, j - 2))
      if i + 2 < n and j + 2 < m:
        neighbors.append((i + 2, j + 2))

  return neighbors

def get_position_coef(i, j, n, m, order):

  if (i == 0 and j == 0) or (i == 0 and j == m-1) or (i == n-1 and j == 0) or (i == n-1 and j == m-1):
        position_type = 'corner'
  elif i == 0 or i == n-1 or j == 0 or j == m-1:
        position_type = 'side'
  else:
        position_type = 'inner'


  table = {'corner': [2, 3, 5,  7,   8],
           'side':   [2, 5, 8,  12, 14],
           'inner':  [4, 8, 12, 20, 24]}

  coef = table[position_type][order-1]

  return coef


def initialize_Q(shape, order=1):
  n, m = shape
  Q = np.zeros((n, m))
  for i in range(n):
    for j in range(m):
      Q[i, j] = get_position_coef(i, j, n, m, order)


  Q = np.diag(Q.flatten())
  index_mapping = {(i, j): i * m + j for i in range(n) for j in range(m)}

  for i in range(n):
    for j in range(m):
      for neighbor in get_neighbor(i, j, n, m, order):
          # Bias for each interaction term.
          # Future-Work: Add a bias dropoff for higher order.
          Q[index_mapping[(i, j)]][index_mapping[neighbor]] = -1

  return Q

def get_Q(X_noisy, l=0.5):
  """
  Input:
    X_noisy: np.array - Noisy training image
    l: float - retention parameter

  Output:
    Q: np.array - QUBO matrix
  """
  n, m = X_noisy.shape

  Q = initialize_Q((n, m))
  # Q = np.zeros((n*m, n*m))

  Q = abs(1-l) * Q + l * np.diag(-2 * X_noisy.flatten() + 1)

  return Q

def get_cqm(X_noisy, l=0.5, order=1, precision=8):
  """
  Construct a CQM for RMF Image Denoising
  Input:
    X_noisy: np.array - Noisy training image
    l: float - retention parameter
    precision: int - upper bound of pixel range

  Output:
    cqm: CQM model
  """
  n, m = X_noisy.shape
  sum_intensity = np.sum(X_noisy)
  index_mapping = {(i, j): i * m + j for i in range(n) for j in range(m)} # Maps 2D indices to 1D indices

  # List of Integers that form the reconstructed image
  X_recon = [Integer(f'X_{i:03}_{j:03}', lower_bound=0, upper_bound=precision) for i in range(n) for j in range(m)]
  cqm = CQM()

  objective = quicksum(
    (1 - l) * (
        quicksum( X_recon[index_mapping[(i, j)]]**2 + X_recon[index_mapping[neighbor]]**2
                 - 2 * X_recon[index_mapping[neighbor]] * X_recon[index_mapping[(i, j)]]
                  for neighbor in get_neighbor(i, j, n, m, order))  # Neighborhood Distance
    )
    + l * (
        X_recon[index_mapping[(i, j)]] ** 2
        - 2 * X_recon[index_mapping[(i, j)]] * X_noisy[i, j] + X_noisy[i, j] ** 2 # Retention
    )

    for i in range(n) for j in range(m)
  )

  cqm.set_objective(objective)

  return cqm

def get_energy(X_noisy, solution, l):
  n, m = X_noisy.shape

  energy = 0
  for i in range(n):
    for j in range(m):
      for neighbor in get_neighbor(i, j, n, m):
        energy += solution[i, j]**2 + solution[neighbor]**2 - 2 * solution[neighbor] * solution[(i, j)]

      energy += l * ( solution[(i, j)] ** 2
          - 2 * solution[(i, j)] * X_noisy[i, j]) + X_noisy[i, j] ** 2

  return energy

def sample_cqm(cqm, simulate=True, **kwargs):
  if simulate:
    bqm, invert = cqm_to_bqm(cqm)
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, **kwargs).aggregate()
    solution = np.array(list(invert(sampleset.first.sample).values())).reshape(tuple(int(x)+1 for x in cqm.variables[-1].split('_')[1:]))
  else:
    sampler = LeapHybridCQMSampler(token=solver_token)
    sampleset = sampler.sample_cqm(cqm, label='RMF Real', **kwargs)
    solution = np.array(list(sampleset.first[0].values())).reshape(tuple(int(x)+1 for x in cqm.variables[-1].split('_')[1:]))

  return solution

