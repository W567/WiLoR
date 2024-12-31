from setuptools import setup, find_packages

setup(
    name='wilor',  # The name of your package
    version='0.1.0',  # Version number
    packages=find_packages(),  # Automatically find all the sub-packages
    install_requires=[  # The packages required for your package
        'numpy',
        'opencv-python',
        'pyrender',
        'pytorch-lightning',
        'scikit-image',
        'smplx==0.1.28',
        'yacs',
        'chumpy @ git+https://github.com/mattloper/chumpy',
        'timm',
        'einops',
        'xtcocotools',
        'pandas',
        'hydra-core',
        'hydra-submitit-launcher',
        'hydra-colorlog',
        'pyrootutils',
        'rich',
        'webdataset',
        'gradio',
        'dill',
        'ultralytics==8.1.34',
    ],
    entry_points={  # Optional: If you want to define command line tools
        'console_scripts': [
            'wilor=wilor.cli:main',  # Assuming you have a cli.py with a main() function
        ],
    },
    author='Rolandos Alexandros Potamias',  # Your name or organization
    author_email='your.email@example.com',  # Your email
    description='Wilor package for hand reconstruction',  # Description of your package
    long_description=open('README.md').read(),  # If you have a README file for a long description
    long_description_content_type='text/markdown',  # Format for the README (markdown, plain text, etc.)
    url='https://github.com/rolpotamias/WiLoR',  # URL of the package (e.g., your GitHub repo)
    classifiers=[  # Optional: Categorize your package (can help discoverability)
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)

