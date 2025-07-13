# rrepps
## Revisiting representer point selection for interpretable predictions

Python code for the experiments described in the paper "Revisiting representer point selection for interpretable predictions" submitted to ICTAI 2025.

### Convexity test

The convexity tests can be run simply by
```
python cvx.py
```
The code uses the datasets in the `data/gaussian` folder, specifically `gaussian_train_100_10` and `gaussian_val_100_10` (currently, only the training set is used).
The json files in the above-mentioned directory are copies of the datasets in JSON format.

### Training and testing vanilla NN models

For training and testing ResNet models (currently implemented), `baseline.py` has to be used either with the `--train` or the `--test` arguments. 
Every parameter (e.g. model used, for example ResNet-50, learning rate, training and test image folders, etc.) is currently controlled via `settings.py`. 
The description of the available parameters will be given below.

Models are written to the `models` folder using the following naming format: `model_best_<acc/loss>_<model_name>.pth`, 
where `<acc/loss>` becomes *acc* and *loss* meaning the best accuracy and best loss models, while `<model_name>` is the name of the model, for example *resnet18*.

When running the program the first time, two files are created in the `data` subfolder: `trainset` and `valset`. Any subsequent run will use these files containing the train/validation split of the data in 80%/20% ratio (hard-coded). If you don't want to use the same split in the next runs, delete these before running the script.

### Training the XSVM model using Pegasos

The XSVM can be trained and tested using the `xsvm.py` script. 
Similarly to the above, it has the following arguments:
* `--train` - trains the SVM;
* `--test` - test the SVM;
* `--vis <filename>` - visualizes the 9 most influential positive and negative training examples.

The script produces a model file in the `models` subfolder containing the weights of the classifier, the alpha expansion coefficients and a dictionary mapping training set filenames to indices.
The name of this file is formatted as follows: `xsvm_<model_name>_<id>.pickle`, where `<id>` can theoretically be any kind of ID, however, currently is set to 0 in the code.
When testing, this model should be in the `models` folder in order to run the test.

(Note: While there are 3 different implementations of Pegasos, the code currently uses the pytorch-based version.) 

### Parameters

The parameters are stored and thus can be set via `settings.py` and are the following (we are giving here a complete settings file explaining each parameter in the comments):

```
NUM_CLASSES = 10 # number of classes
DATA_PATH_TRAIN = 'path_to_train_images' # the images has to be in different subfolders corresponding to classes
DATA_PATH_TEST = 'path_to_test_images'   # the same as above applies

DATA_PERCENTAGE = 1.0 # percentage of data to be used; for fast experimentation one can lower this value

BATCH_SIZE = 64 # batch size when training the NN
BATCH_SIZE_XSVM = 32 # batch size in Pegasos
NUM_EPOCHS = 100 # number of epochs when training the NN
PATIENCE = 20 # patience; currently StepLR is used with step_size=10 and gamma=0.1
LR = 1e-2 # learning rate of the optimizer; currently SGD is used
MOMENTUM = 0.9 # momentum of SGD
NUM_SAMPLES_COEF = 1 # the number of iterations of Pegasos is NUM_SAMPLES_COEF * n, where n is the number of training examples

TRAINSET = 'data/trainset' # output name and location of training set
VALSET = 'data/valset'     # output name and location of validation set

MODEL_NAME = 'resnet50' # model name; currently resnet18, resnet34 and resnet50 are available
BEST_MODEL = 'acc' # acc or loss; which best model to use when testing or training XSVM

AUGMENT_XSVM = False # whether use augmentation or not when training the SVM

IMG_MEAN = [0.4761, 0.4518, 0.3910] # mean for image normalization
IMG_STD  = [0.2580, 0.2525, 0.2571] # standard deviation for image normalization

CUDA_ID = 0 # ID of the GPU with CUDA support
```
