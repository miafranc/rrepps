# rrepps
## Revisiting representer point selection for interpretable predictions

Python code for the experiments described in the paper "Revisiting representer point selection for interpretable predictions".

We also uploaded the trained ResNet-50(-based) models for the Stanford Dogs dataset, therefore all the tests and visualizations should work for this dataset and model.

### Convexity test

The convexity tests can be run simply by
```
python cvx.py
```
The code uses the datasets in the `data/gaussian` folder, specifically `gaussian_train_100_10` and `gaussian_val_100_10` (currently, only the training set is used).
The json files in the above-mentioned directory are copies of the datasets in JSON format.

### Training and testing vanilla CNNs

CNNs can be trained, tested, and SVM-based CNNs can be fine-tuned via `baseline.py`.

For training and testing CNNs; currently implemented: ResNet-34, ResNet-50, DenseNet-121, DenseNet-161, ConvNeXt-T.
Every parameter (e.g. model used, learning rate, training and test image folders, etc.) is controlled via `settings.py`. 
The description of the available parameters will be given below.

Command-line arguments:
* `--train` - trains the CNN;
* `--test` - tests the model;
* `--svm` - if used with `--test`, tests the SVM;
* `--finetuned` - if used with `--train`, fine-tunes the model; if used with `--test`, uses the fine-tuned SVM.

Corruptions can also be applied to the test images via `settings.py`.

### The XSVM model

The XSVM can be trained and tested using the `xsvm.py` script. 

Command-line arguments:
* `--generate` - generates the features for training;
* `--train` - trains the SVM;
* `--test` - tests the SVM;
* `--build_nn` - builds the neural network using the SVM's classifier weights; it is needed for performing the decomposed Grad-CAM, and to perform fine-tuning.
* `--visualize FILENAME` -- visualizes the 9 most important positive examples (if negative are needed, it has to be changed in the code);
* `--visualize_gradcam FILENAME` -- visualizes the 9 most important positive training examples showing the Grad-CAM saliency.

Only the above options are reachable through command-line arguments, but everything described in the paper can be performed (e.g. calculating layer divesity) from code.

### Parameters

The parameters are stored and thus can be set via `settings.py` and are the following (we are giving here a complete settings file explaining each parameter in the comments, except where the name makes it clear, what the respective parameter is for):

```
NUM_CLASSES = 120 # number of classes
DATASET_NAME = 'dataset_name' # name of dataset; name of the subfolder on DATA_PATH

DATA_PATH = 'path_to_train_images' # the images has to be in different subfolders corresponding to classes; main folder: train and test
DATA_FEATURES_PATH = 'path_to_storing_feature_outputs'   # the same as above applies

DATA_PERCENTAGE = 1.0 # percentage of data to be used; for fast experimentation one can lower this value
VAL_SPIT = 0.2

BATCH_SIZE = 64
NUM_EPOCHS = 100
PATIENCE = 20
OPTIMIZER = 'sgd'
LR = 1e-3
MOMENTUM = 0.9 # momentum of SGD
WEIGHT_DECAY = 1e-4 # weight decay of optimizer, if applicable

NUM_EPOCHS_FINETUNE = 5 # number of fine-tuning epochs
LR_FINETUNE = 1e-4 # LR when fine-tuning

SCHEDULER = 'cosine'
STEP_SIZE = 10 # for StepLR
STEP_GAMMA = 0.1 # for StepLR and MultiStepLR
STEP_MULTI = [30, 60, 90] # for MultiStepLR

MODEL_NAME = 'resnet50'
BASE_MODEL_BIAS = True # wheather use bias or not
BEST_MODEL = 'acc' # which perspective is considered (acc or loss)
MODEL_PATH = 'models/' # where models are saved to and loaded from
SAVE_BEST_MODEL = True # wheather to save the best model
SAVE_MIN_ACC = 0.7 # accuracy threshold above which models are saved

IMG_SIZE = 224
IMG_MEAN = [0.4761, 0.4518, 0.3910] # mean for image normalization
IMG_STD  = [0.2580, 0.2525, 0.2571] # standard deviation for image normalization

TENSORBOARD_LOGDIR_PREFIX = 'runs/' # tensorboard log directory

CUDA_ID = 0 # ID of GPU

CORRUPTION = None # corruption type and parameters; see `image_dataset.py` for corruption types
CORRUPTION_BLUR_RADIUS = 2
CORRUPTION_NOISE_STD = 20
CORRUPTION_NOISE_BLEND = 30
CORRUPTION_CONTRAST = 1.5
CORRUPTION_BRIGHTNESS = 1.2
CORRUPTION_PIXELATE_LEVEL = 2
```

### Datasets used in the experiments

* CUB-200-2011: [https://www.vision.caltech.edu/datasets/cub_200_2011/](https://www.vision.caltech.edu/datasets/cub_200_2011/)
* Stanford Dogs: [http://vision.stanford.edu/aditya86/ImageNetDogs/](http://vision.stanford.edu/aditya86/ImageNetDogs/)
* Stanford Cars: [https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder](https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder) ([https://docs.pytorch.org/vision/main/generated/torchvision.datasets.StanfordCars.html](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.StanfordCars.html))