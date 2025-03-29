from setuptools import setup, find_packages

setup(
    name="cascade2p",
    version="1.0",
    description="Calibrated inference of spiking from calcium ΔF/F data using deep networks",
    author="Peter Rupprecht",
    author_email="",
    packages=find_packages(),
    python_requires=">=3.8, <3.9", # changed this line
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "tensorflow",  # pip install CPU and GPU tensorflow
        "h5py",
        "seaborn",
        "ruamel.yaml",
        "mat73"
    ],
)
