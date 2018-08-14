"""
Module for running predictions with fast ai model
"""
from fastai.conv_learner import *
from fastai.models.cifar10.resnext import resnext29_8_64



def get_data(sz,bs):
    tfms = tfms_from_stats(stats, sz, aug_tfms=[RandomFlip()], pad=sz//8)
    return ImageClassifierData.from_paths(PATH, val_name='test', tfms=tfms, bs=bs)





if __name__ == '__main__':
    PATH = "data/"
    os.makedirs(PATH, exist_ok=True)
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    stats = (np.array([0.4914, 0.48216, 0.44653]), np.array([0.24703, 0.24349, 0.26159]))

    data = get_data(32, 32)  # data generator for batch size=64, image size=64x64

    m = resnext29_8_64()
    bm = BasicModel(m.cuda(), name='cifar10_rn29_8_64')

    learn = ConvLearner(data, bm)
    learn.load("32x32_8")

    trn_tfms, val_tfms = tfms_from_model(m, 32)  # get transformations
    im = val_tfms(open_image('./data/test/8/758.jpg'))
    learn.precompute = False  # We'll pass in a raw image, not activations
    preds = learn.predict_array(im[None])
    np.argmax(preds)  # preds are log probabilities of classes