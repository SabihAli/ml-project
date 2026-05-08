from setuptools import setup, find_packages

setup(
    name="nbjw_calib",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "torch",
        "torchvision",
        "pyyaml",
        "scipy",
    ],
)
