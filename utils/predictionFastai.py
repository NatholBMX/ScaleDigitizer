"""
Module for running predictions with fast ai model
"""
from fastai.conv_learner import *
from fastai.models.cifar10.resnext import resnext29_8_64

PATH = "data/"
CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
STATS = (np.array([0.4914, 0.48216, 0.44653]), np.array([0.24703, 0.24349, 0.26159]))

IMAGE_SIZE = 32
BATCH_SIZE = 32

model = None


def get_data(sz, bs):
    tfms = tfms_from_stats(STATS, sz, aug_tfms=[RandomFlip()], pad=sz // 8)
    return ImageClassifierData.from_paths(PATH, val_name='test', tfms=tfms, bs=bs)


def init_model():
    global model
    os.makedirs(PATH, exist_ok=True)
    data = get_data(BATCH_SIZE, IMAGE_SIZE)  # data generator for batch size=32, image size=32x32

    mod = resnext29_8_64()
    basemodel = BasicModel(mod.cuda(), name='cifar10_rn29_8_64')
    model = ConvLearner(data, basemodel)

    model.load("32x32_8")


def predict_from_model(image):
    trn_tfms, val_tfms = tfms_from_model(resnext29_8_64(), IMAGE_SIZE)  # get transformations
    # im = val_tfms(open_image('./data/test/8/758.jpg'))
    im = val_tfms(image)
    model.precompute = False  # We'll pass in a raw image, not activations
    preds = model.predict_array(im[None])
    prediction = np.argmax(preds)  # preds are log probabilities of CLASSES

    return prediction


if __name__ == '__main__':
    init_model()
    prediction = predict_from_model(open_image('./data/test/8/758.jpg'))
    print(prediction)
