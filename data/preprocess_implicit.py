import zipfile
import os
import os.path as path
import csv
import numpy as np
from scipy import sparse
import cPickle as pickle

#=============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', default="ml-1m", type=str,
        help="Which dataset to process")

    args = parser.parse_args()

#-----------------------------------------------------------------------------
    if args.dataset == 'ml-1m':
        data_path = 'MovieLens/ml-1m/'
        if not path.exists(data_path):
            zip_ref = zipfile.ZipFile('MovieLens/zipfiles/' + args.dataset + '.zip', 'r')
            zip_ref.extractall('MovieLens/')
            zip_ref.close()
            print 'unzipped files.'
        pickled_dir = path.join(data_path, 'pickled_data')
        if not path.exists(pickled_dir):
            os.makedirs(pickled_dir)

        N_u, N_v = 6040, 3952 # according to the README
        print '%d users\n%d items'%(N_u, N_v)

        with open(data_path + 'ratings.dat') as f:
            reader = csv.reader(f, delimiter=':')
            d = list(reader)

        R = np.array(d)
        R = R[:, [0,2,6]]
        R = R.T
        R = R.astype('int64')
        R[0] = R[0] - 1
        R[1] = R[1] - 1

        T_matrix = sparse.coo_matrix((R[2], (R[0], R[1])), shape=(N_u, N_v)).toarray()
        R_matrix = (T_matrix > 0).astype('int32')
        assert R.shape[1] == np.count_nonzero(R_matrix)

        R_latest     = np.argsort(T_matrix)[:, -1].astype('int32')
        R_2nd_latest = np.argsort(T_matrix)[:, -2].astype('int32')

        assert np.all(T_matrix[np.arange(N_u), R_latest] > 0)
        assert np.all(T_matrix[np.arange(N_u), R_2nd_latest] > 0)
        assert np.all(R_matrix[np.arange(N_u), R_latest] == 1)
        assert np.all(R_matrix[np.arange(N_u), R_2nd_latest] == 1)

        pickled_file = path.join(pickled_dir, 'implicit.pkl')
        with open(pickled_file, 'wb') as pklf:

            data={}
            data['T_matrix'] = T_matrix
            data['R_matrix'] = R_matrix
            data['N_interaction'] = R.shape[1]
            data['R_latest'] = R_latest
            data['R_2nd_latest'] = R_2nd_latest
            data['N_users'] = N_u
            data['N_items'] = N_v

            print '%d interactions' % data['N_interaction']

            pickle.dump(data, pklf)
            print 'pickled file saved to %s\nDone.' % pickled_file

#-----------------------------------------------------------------------------
    elif args.dataset == 'pinterest':
        pass # don't need this dataset any more

#-----------------------------------------------------------------------------
    elif args.dataset == 'ml-10m':
        pass # don't need this dataset any more

#-----------------------------------------------------------------------------
    else:
        print "Wrong dataset!"

    print 'implicit %s preprocessing finished.' % args.dataset
