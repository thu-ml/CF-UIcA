import zipfile
import os
import os.path as path
import csv
import numpy as np
from scipy import sparse
import cPickle as pickle

def read_csv_to_R(fn, delimiter, rating_scale=1, is_double_ch_delimiter=False):
    with open(fn) as f:
        reader = csv.reader(f, delimiter=delimiter)
        d = list(reader)

        if is_double_ch_delimiter:
            R = np.array(d).astype('string')
            R = R[:, [0,2,4]]
            R = R.astype('float32')
        else:
            R = np.array(d).astype('float32')
            R = R[:,0:3]

        R = R.T
        R[2] = R[2] * rating_scale
        R = R.astype('int32')
        R[0] = R[0] - 1
        R[1] = R[1] - 1

        return R

#=============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', default="ml-1m", type=str,
        help="Which dataset to process")

    parser.add_argument('-n_splits', default=10, type=int,
        help="# of train-test splits to be generated")

    parser.add_argument('-seed', default=12345, type=int,
        help="Random seed")

    args = parser.parse_args()

#-----------------------------------------------------------------------------
    if args.dataset == 'ml-100k':
        pass # don't need this dataset any more

#-----------------------------------------------------------------------------
    elif args.dataset == 'ml-1m':
        data_path = 'MovieLens/ml-1m/'
        if not path.exists(data_path):
            zip_ref = zipfile.ZipFile('MovieLens/zipfiles/' + args.dataset + '.zip', 'r')
            zip_ref.extractall('MovieLens/')
            zip_ref.close()
            print 'unzipped files.'
        pickled_dir = path.join(data_path, 'pickled_data')
        if not path.exists(pickled_dir):
            os.makedirs(pickled_dir)

        R = read_csv_to_R(data_path+'ratings.dat', ':', is_double_ch_delimiter=True)
        N_u, N_v = 6040, 3952 # according to the README
        print '%d users\n%d items'%(N_u,N_v)

        # create train-test splits
        random_state = np.random.RandomState(args.seed)
        rand_perm = random_state.permutation(R.shape[1])
        splits = np.array_split(rand_perm, args.n_splits)

        for sid in range(args.n_splits):
            train_ids = np.concatenate(splits[:sid] + splits[sid+1:])
            test_ids = splits[sid]
            R_train = R[:, train_ids]
            R_test = R[:, test_ids]
            assert R_train.shape[1] + R_test.shape[1] == R.shape[1]

            # save train-test split
            pickled_file = path.join(pickled_dir, 'split-%d.pkl'%(sid+1))
            with open(pickled_file, 'wb') as pklf:

                data={}
                data['R_train'] = R_train
                data['N_train_rating'] = R_train.shape[1]
                data['R_test'] = R_test
                data['N_test_rating'] = R_test.shape[1]
                data['N_users'] = N_u
                data['N_items'] = N_v
                data['N_stars'] = 5

                print '%d train ratings\n%d test ratings'%(data['N_train_rating'],data['N_test_rating'])

                pickle.dump(data, pklf)
                print 'pickled file saved to %s\nDone.' % pickled_file

#-----------------------------------------------------------------------------
    elif args.dataset == 'ml-10m':
        pass # don't need this dataset any more

#-----------------------------------------------------------------------------
    elif args.dataset == 'ml-20m':
        pass # don't need this dataset any more

#-----------------------------------------------------------------------------
    else:
        print "Wrong dataset!"

    print '%s preprocessing finished.' % args.dataset
