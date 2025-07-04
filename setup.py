from setuptools import setup, find_packages


# Read requirements from requirements.txt
def read_requirements():
    with open("mtb/requirements.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="mtb",
    version="0.1.0",
    description="Mountain Bike Data Tracker and Visualizer",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Evan Sims",
    author_email="easims@gmail.com",
    url="https://github.com/es65/MtbViz",
    packages=find_packages(),
    install_requires=read_requirements(),
    python_requires=">=3.8",
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "viz=mtb.app.app_vis:main",
            "process=mtb.src.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="mountain biking, data analysis, sensor data, gps, accelerometer",
    project_urls={
        "Bug Reports": "https://github.com/es65/MtbViz/issues",
        "Source": "https://github.com/es65/MtbViz",
        "Documentation": "https://github.com/es65/MtbViz#readme",
    },
)

"""
Installation options:

1. For development (editable install):
   pip install -e .

2. For production:
   pip install .

3. For dedicated venv (recommended):
   cd mtb/
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
"""
