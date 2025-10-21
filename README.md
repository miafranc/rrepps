# rrepps
## Revisiting representer point selection for interpretable predictions

Python code for the experiments described in the paper "Revisiting representer point selection for interpretable predictions".

### Convexity test

The convexity tests can be run simply by
```
python cvx.py
```
The code uses the datasets in the `data/gaussian` folder, specifically `gaussian_train_100_10` and `gaussian_val_100_10` (currently, only the training set is used).
The json files in the above-mentioned directory are copies of the datasets in JSON format.

### Training and testing vanilla CNNs

For training and testing CNNs (currently implemented: ResNet-34, ResNet-50, DenseNet-121, DenseNet-161, ConvNeXt-T), `baseline.py` has to be used either with the `--train` or the `--test` arguments. 
Every parameter (e.g. model used, learning rate, training and test image folders, etc.) is controlled via `settings.py`. 
The description of the available parameters will be given below.

...

### The XSVM model

The XSVM can be trained and tested using the `xsvm.py` script. 
Similarly to the above, it has the following arguments:
* `--train` - trains the SVM;
* `--test` - test the SVM;
* `--vis <filename>` - visualizes the 9 most influential positive and negative training examples.

...

...

### Parameters

The parameters are stored and thus can be set via `settings.py` and are the following (we are giving here a complete settings file explaining each parameter in the comments):

```
NUM_CLASSES = 120 # number of classes
DATASET_NAME = 'dataset_name' # name of dataset; name of the subfolder on DATA_PATH

DATA_PATH = 'path_to_train_images' # the images has to be in different subfolders corresponding to classes; main folder: train and test
DATA_FEATURES_PATH = 'path_to_storing_feature_outputs'   # the same as above applies

DATA_PERCENTAGE = 1.0 # percentage of data to be used; for fast experimentation one can lower this value
VAL_SPIT = 0.2 # validation split

BATCH_SIZE = 64 # batch size when training the NN
NUM_EPOCHS = 100 # number of epochs when training the NN
PATIENCE = 20 # patience; currently StepLR is used with step_size=10 and gamma=0.1
OPTIMIZER = 'sgd' # optimizer
LR = 1e-3 # learning rate of the optimizer
MOMENTUM = 0.9 # momentum of SGD
WEIGHT_DECAY = 1e-4 # weight decay of optimizer, if applicable

NUM_EPOCHS_FINETUNE = 5
LR_FINETUNE = 1e-4

SCHEDULER = 'cosine'
STEP_SIZE = 10
STEP_GAMMA = 0.1
STEP_MULTI = [30, 60, 90]

MODEL_NAME = 'resnet50'
BASE_MODEL_BIAS = True
BEST_MODEL = 'acc'
MODEL_PATH = 'models/'
SAVE_BEST_MODEL = True
SAVE_MIN_ACC = 0.7

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
* Stanford Cars: [https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder](https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder)