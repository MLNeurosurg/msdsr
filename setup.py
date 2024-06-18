from setuptools import setup, find_packages

setup(name="msdsr",
      version="1.0",
      packages=find_packages(),
      install_requires=[
          "setuptools>=60.7.0",
          "pip>=22.0.0",
          "torchvision>=0.16.0",
          "torch>=2.1.0",
          "pytorch-lightning>=2.1.1",
          "tensorboard",
          "pytest",
          "yapf",
          "tqdm",
          "scikit-learn",
          "opencv-python>=4.2.0.34",
          "jupyterlab",
          'gpustat',
          'openpyxl',
          'pandas',
          'tifffile',
          'pydicom',
          'zipfile38',
          'imageio',
          'scikit-image',
          "pyyaml",
          "matplotlib",
          "altair",
          "nibabel",
          "monai",
          "timm",
          "ijson",
          "gpustat",
          "einops",
          "torch-fidelity"
      ],
      dependency_links=['https://download.pytorch.org/whl/cu116'])