from setuptools import setup, find_packages

setup(
    name='RSNA_torch',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['numpy','pandas', 'torchvision', 'torch', 'pytorch_lightning', 'hydra', "python-gdcm", "pydicom", "pylibjpeg"], # list of dependencies
    # other options
)