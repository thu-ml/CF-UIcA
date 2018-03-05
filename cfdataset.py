from __future__ import division

import os
import logging
import cPickle as pickle
import os.path as path

import numpy as np
from scipy import sparse

logger = logging.getLogger()

#-----------------------------------------------------------------------------

def datapath(fname):
    """ Try to find *fname* in the dataset directory and return
        a absolute path.
    """
    candidates = [
        path.abspath("."),
        path.abspath("data"),
    ]
    if 'DATASET_PATH' in os.environ:
        candidates.append(os.environ['DATASET_PATH'])

    for c in candidates:
        c = path.join(c, fname)
        if path.exists(c):
            return c

    raise IOError("Could not find %s" % fname)


#-----------------------------------------------------------------------------

class CFdata(object):
    def __init__(self, name='ml-1m', split=1):

        CFdatasets = {
                #'ml-100k': 'MovieLens/ml-100k/pickled_data/',
                'ml-1m': 'MovieLens/ml-1m/pickled_data/',
                #'ml-10m': 'MovieLens/ml-10M100K/pickled_data/',
                #'netflix': 'Netflix/pickled_data/',
            }
        assert name in CFdatasets.keys()

        logger.info("Loading %s data, split: %d" % (name, split))

        fname = datapath(CFdatasets[name]+'split-'+str(split)+'.pkl')
        try:
            with open(fname, 'rb') as f:
                data = pickle.load(f)
                self.R_train = data['R_train']
                self.R_test = data['R_test']
                self.N_train_rating= data['N_train_rating']
                self.N_test_rating= data['N_test_rating']
                self.N_users = data['N_users']
                self.N_items = data['N_items']
                self.N_stars = data['N_stars']

                assert self.N_train_rating == self.R_train.shape[1]
                assert self.N_test_rating == self.R_test.shape[1]

        except IOError, e:
            logger.error("Failed to open %s: %s" % (fname, e))
            exit(1)

#-----------------------------------------------------------------------------

class CF_implicit_data(object):
    def __init__(self, name='ml-1m'):

        CF_implicit_datasets = {
                'ml-1m': 'MovieLens/ml-1m/pickled_data/',
                #'pinterest': 'Pinterest/pickled_data/',
            }
        assert name in CF_implicit_datasets.keys()

        logger.info("Loading implicit %s data" % name)

        fname = datapath(CF_implicit_datasets[name]+'implicit.pkl')
        try:
            with open(fname, 'rb') as f:
                data = pickle.load(f)
                self.T_matrix = data['T_matrix']
                self.R_matrix = data['R_matrix']
                self.R_latest = data['R_latest']
                self.R_2nd_latest = data['R_2nd_latest']
                self.N_interaction = data['N_interaction']
                self.N_users = data['N_users']
                self.N_items = data['N_items']

                assert self.N_interaction == np.count_nonzero(self.R_matrix)
                assert self.R_latest.shape[0] == self.N_users
                assert self.R_2nd_latest.shape[0] == self.N_users
                assert np.all(self.T_matrix[np.arange(self.N_users), self.R_latest] > 0)
                assert np.all(self.T_matrix[np.arange(self.N_users), self.R_2nd_latest] > 0)
                assert np.all(self.R_matrix[np.arange(self.N_users), self.R_latest] == 1)
                assert np.all(self.R_matrix[np.arange(self.N_users), self.R_2nd_latest] == 1)

        except IOError, e:
            logger.error("Failed to open %s: %s" % (fname, e))
            exit(1)
