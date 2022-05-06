from typing import Dict, List, Union
import numpy as np

from naslib.predictors.utils.encodings import encode
from naslib.predictors.predictor import Predictor


class BaseTree(Predictor):

    def __init__(self, encoding_type='adjacency_one_hot', ss_type='nasbench201', zc=False, zc_only=False, hpo_wrapper=False):
        super(Predictor, self).__init__()
        self.encoding_type = encoding_type
        self.ss_type = ss_type
        self.zc = zc
        self.zc_names = None
        self.zc_only = zc_only
        self.hyperparams = None
        self.hpo_wrapper = hpo_wrapper

    @property
    def default_hyperparams(self):
        return {}

    def get_dataset(self, encodings, labels=None):
        return NotImplementedError('Tree cannot process the numpy data without \
                                   converting to the proper representation')

    def train(self, train_data, **kwargs):
        return NotImplementedError('Train method not implemented')

    def predict(self, data, **kwargs):
        return self.model.predict(data, **kwargs)

    def fit(self, xtrain, ytrain, train_info=None, params=None, **kwargs):

        # normalize accuracies
        self.mean = np.mean(ytrain)
        self.std = np.std(ytrain)

        if type(xtrain) is list:
            # when used in itself, we use
            xtrain = np.array([encode(arch, encoding_type=self.encoding_type,
                                      ss_type=self.ss_type) for arch in xtrain])

            if self.zc:
                # mean, std = -10000000.0, 150000000.0
                # xtrain = [[*x, (train_info[i]-mean)/std] for i, x in enumerate(xtrain)]
                if self.zc_only:
                    xtrain = self.zc_features
                else:
                    xtrain = [[*x, *zc_scores] for x, zc_scores in zip (xtrain, self.zc_features)]
            xtrain = np.array(xtrain)
            ytrain = np.array(ytrain)

        else:
            # when used in aug_lcsvr we feed in ndarray directly
            xtrain = xtrain
            ytrain = ytrain


        # convert to the right representation
        train_data = self.get_dataset(xtrain, ytrain)

        # fit to the training data
        self.model = self.train(train_data)

        # predict
        train_pred = np.squeeze(self.predict(xtrain))
        train_error = np.mean(abs(train_pred-ytrain))

        return train_error

    def query(self, xtest, info=None):

        if type(xtest) is list:
            #  when used in itself, we use
            xtest = np.array([encode(arch, encoding_type=self.encoding_type,
                                 ss_type=self.ss_type) for arch in xtest])
            if self.zc:
                # mean, std = -10000000.0, 150000000.0
                zc_scores = [self.create_zc_feature_vector(data['zero_cost_scores']) for data in info]
                if self.zc_only:
                    xtest = zc_scores
                else:
                    xtest = [[*x, *zc] for x, zc in zip(xtest, zc_scores)]
            xtest = np.array(xtest)

        else:
            # when used in aug_lcsvr we feed in ndarray directly
            xtest = xtest

        test_data = self.get_dataset(xtest)
        return np.squeeze(self.model.predict(test_data)) * self.std + self.mean

    def get_random_hyperparams(self):
        pass

    def create_zc_feature_vector(self, zero_cost_scores: Union[List[Dict], Dict]) -> Union[List[List], List]:
        zc_features = []

        def _make_features(zc_scores):
            zc_values = []
            for zc_name in self.zc_names:
                zc_values.append(zc_scores[zc_name])

            zc_features.append(zc_values)

        if isinstance(zero_cost_scores, list):
            for zc_scores in zero_cost_scores:
                _make_features(zc_scores)
        elif isinstance(zero_cost_scores, dict):
            _make_features(zero_cost_scores)
            zc_features = zc_features[0]

        return zc_features

    def set_hyperparams(self, params):
        self.hyperparams = params