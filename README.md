# Quantum-RMF

Quantum-RMF is a research-oriented project that performs image denoising using quantum annealing. The task is framed as a Random Markov Field (RMF) combinatorial optimization problem. This repository contains code implementations, utilities, and experiment results for binary and discrete RMF models, showcasing how quantum annealing techniques can optimize RMFs for effective noise reduction in images.

## Overview

Image denoising is a critical task in fields such as computer vision, medical imaging, and remote sensing, where the removal of noise is essential to improve the clarity and accuracy of images. Random Markov Fields (RMFs) offer a robust probabilistic framework to model spatial dependencies in images, making them highly suitable for denoising tasks. RMFs balance data fidelity and spatial smoothness through an energy function that captures pixel interactions, and the optimization of this function enables effective noise reduction.

The Quantum-RMF project investigates quantum-inspired optimization approaches for RMFs, focusing on:

- Binary RMFs, where pixel values are binary (0 or 1).
- Discrete RMFs, where pixel values can take discrete integer values.

The repository demonstrates how RMFs can be used to reconstruct and denoise images while preserving essential details such as edges and textures. The work is particularly relevant for quantum annealing platforms, as the RMF optimization problem can be formulated as a Quadratic Unconstrained Binary Optimization (QUBO) problem.

For detailed results and a comprehensive discussion, refer to the [full thesis document](https://drive.google.com/file/d/1iulbGETJ4FszEk_l6DG0w3Y6OI0PTJxa/view?usp=sharing).


## Installation

To get started with the project, clone the repository and install the required dependencies:

```
git clone https://github.com/your-username/Quantum-RMF.git
cd Quantum-RMF
pip install -r requirements.txt
```

## Usage

The main demonstrations of the models are provided in the Jupyter notebooks located in the root directory:

- `RMF-binary.ipynb`: Demonstrates the binary RMF model, which formulates the optimization problem as a QUBO.
- `RMF-discrete.ipynb`: Demonstrates the discrete RMF model, utilizing constrained quadratic models (CQM) for optimization.

The utility scripts (`binary.py` and `discrete.py`) contain helper methods for performing RMF computations and are used by the notebooks.

### Running the Jupyter Notebooks

Start the Jupyter Notebook server:

```
jupyter notebook
```

Open the desired notebook (RMF-binary.ipynb or RMF-discrete.ipynb) to explore binary or discrete RMF demonstrations. Modify the configuration parameters in the notebooks to adjust model settings, such as:

- Retention parameter (𝜆) for balancing data fidelity and spatial smoothness.
- Neighborhood order (𝑜) for controlling pixel connectivity.

### Experiment Highlights
- Binary RMF: Uses binary pixel values (0 or 1) and applies QUBO formulations for optimization. Suitable for thresholded grayscale images.
- Discrete RMF: Extends the binary case to handle discrete pixel values, using a CQM approach to encode the problem for quantum annealing.

## Results

Experiment results are saved in the `results/` directory, organized by model type (binary or discrete). Saved runs for reproducibility are stored in the `runs/` directory.

### Metrics

The effectiveness of denoising is evaluated using:

- **Structural Similarity Index (SSI)**: Measures similarity between the denoised and ground truth images.
- **Peak Signal-to-Noise Ratio (PSNR)**: Quantifies the ratio between the maximum possible signal and the noise.
- **False Positive and False Negative Counts**: Evaluate pixel-level accuracy in binary reconstructions.
- **L2 Loss**: Measures the squared difference between the reconstructed and ground truth images.

## Contributing

Contributions are welcome! If you'd like to improve this project, feel free to fork the repository, make your changes, and submit a pull request. For major changes, please open an issue first to discuss your ideas.

## License

This project is licensed under the [MIT License](LICENSE).
