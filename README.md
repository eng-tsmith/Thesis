# Master Thesis

## Dependencies
You need a [anaconda](https://www.continuum.io/downloads) or [miniconda](https://conda.io/miniconda.html) to use the environment setting.
You can install all dependencies by running one of the following commands from the '/config' folder:
```python
# TensorFlow without GPU
conda env create -f cpu_keras.yml

# TensorFlow with GPU
conda env create -f gpu_keras.yml
```

Or you can manually install the required libraries (see the contents of the environemnt*.yml files) using pip.


## Usage
### Run the pretrained model

Start up [the Udacity self-driving simulator](https://github.com/udacity/self-driving-car-sim), choose a scene and press the Autonomous Mode button.  Then, run the model as follows:

```python
python drive.py model.h5
```

### To train the model

You'll need the data folder which contains the training images.

```python
python model.py
```
