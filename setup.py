from setuptools import setup, find_packages

setup(
    name='RSNA_torch',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['numpy','pandas', 'torchvision', 'torch', 'pytorch_lightning', 'hydra-core', "python-gdcm", "pydicom", "pylibjpeg", "omegaconf", "focal-loss-torch"], # list of dependencies
    # other options
)
