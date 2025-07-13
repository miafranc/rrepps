NUM_CLASSES = 10
DATA_PATH_TRAIN = '../stanford dogs/small2/train'
DATA_PATH_TEST = '../stanford dogs/small2/test'

DATA_PERCENTAGE = 1.0

BATCH_SIZE = 64
BATCH_SIZE_XSVM = 32
NUM_EPOCHS = 100
PATIENCE = 20
LR = 1e-2
MOMENTUM = 0.9
NUM_SAMPLES_COEF = 1

TRAINSET = 'data/trainset'
VALSET = 'data/valset'

MODEL_NAME = 'resnet50'
BEST_MODEL = 'acc'

AUGMENT_XSVM = False

IMG_MEAN = [0.4761, 0.4518, 0.3910]
IMG_STD  = [0.2580, 0.2525, 0.2571]

CUDA_ID = 0
