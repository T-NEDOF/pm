import numpy as np
import cv2
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn import preprocessing

class TargetEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, encoder=LabelEncoder()):
        self.encoder = encoder

    def fit(self, X, y=None):
        # note that x is the target in this case
        self.encoder.fit(X)
        return self

    def transform(self, X):
        X = X.copy()
        X = np_utils.to_categorical(self.encoder.transform(X))
        return X


def _im_resize(df, n, image_size):
    im = cv2.imread(df[n])
    im = cv2.resize(im, (image_size, image_size))
    return im


class CreateDataset(BaseEstimator, TransformerMixin):

    def __init__(self, image_size=50):
        self.image_size = image_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        tmp = np.zeros((len(X),
                        self.image_size,
                        self.image_size, 3), dtype='float32')

        for n in range(0, len(X)):
            im = _im_resize(X, n, self.image_size)
            tmp[n] = im

        print('Dataset Images shape: {} size: {:,}'.format(
            tmp.shape, tmp.size))
        return tmp

def preprocessing_train_data(train_df):

    # read training data - It is the aircraft engine run-to-failure data

    train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
    train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                        's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                        's15', 's16', 's17', 's18', 's19', 's20', 's21']
    train_df = train_df.sort_values(['id','cycle'])

    # Data Labeling - generate column RUL(Remaining Usefull Life or Time to Failure)
    rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    train_df = train_df.merge(rul, on=['id'], how='left')
    train_df['RUL'] = train_df['max'] - train_df['cycle']
    train_df.drop('max', axis=1, inplace=True)

    # generate label columns for training data
    # we will only make use of "label1" for binary classification, 
    # while trying to answer the question: is a specific engine going to fail within w1 cycles?
    w1 = 30
    w0 = 15
    train_df['label1'] = np.where(train_df['RUL'] <= w1, 1, 0 )
    train_df['label2'] = train_df['label1']
    train_df.loc[train_df['RUL'] <= w0, 'label2'] = 2
    return train_df

def preprocessing_data(test_df):

    # read training data - It is the aircraft engine run-to-failure data

    # train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
    test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                        's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                        's15', 's16', 's17', 's18', 's19', 's20', 's21','RUL']
    test_df = test_df.sort_values(['id','cycle'])

    
    # generate label columns for training data
    # we will only make use of "label1" for binary classification, 
    # while trying to answer the question: is a specific engine going to fail within w1 cycles?
    w1 = 30
    w0 = 15
    test_df['label1'] = np.where(test_df['RUL'] <= w1, 1, 0 )
    test_df['label2'] = test_df['label1']
    test_df.loc[test_df['RUL'] <= w0, 'label2'] = 2
    return test_df

def minmax_normalization(train_df,test_df):
    train_df['cycle_norm'] = train_df['cycle']
    cols_normalize = train_df.columns.difference(['id','cycle','RUL','label1','label2'])
    min_max_scaler = preprocessing.MinMaxScaler()
    norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]), 
                                columns=cols_normalize, 
                                index=train_df.index)
    join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
    train_df = join_df.reindex(columns = train_df.columns)

    test_df['cycle_norm'] = test_df['cycle']
    norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]), 
                                columns=cols_normalize, 
                                index=test_df.index)
    test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
    test_df = test_join_df.reindex(columns = test_df.columns)
    test_df = test_df.reset_index(drop=True)
    return test_df

def minmax_normalization_train(train_df):
    train_df['cycle_norm'] = train_df['cycle']
    cols_normalize = train_df.columns.difference(['id','cycle','RUL','label1','label2'])
    min_max_scaler = preprocessing.MinMaxScaler()
    norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]), 
                                columns=cols_normalize, 
                                index=train_df.index)
    join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
    train_df = join_df.reindex(columns = train_df.columns)
    return train_df

def gen_sequence(id_df, seq_length, seq_cols):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """
    # for one id I put all the rows in a single matrix
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    # Iterate over two lists in parallel.
    # For example id1 have 192 rows and sequence_length is equal to 50
    # so zip iterate over two following list of numbers (0,112),(50,192)
    # 0 50 -> from row 0 to row 50
    # 1 51 -> from row 1 to row 51
    # 2 52 -> from row 2 to row 52
    # ...
    # 111 191 -> from row 111 to 191
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]

def gen_labels(id_df, seq_length, label):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """
    # For one id I put all the labels in a single matrix.
    # For example:
    # [[1]
    # [4]
    # [1]
    # [5]
    # [9]
    # ...
    # [200]] 
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    # I have to remove the first seq_length labels
    # because for one id the first sequence of seq_length size have as target
    # the last label (the previus ones are discarded).
    # All the next id's sequences will have associated step by step one label as target.
    return data_matrix[seq_length:num_elements, :]


class GenSequence(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        seq_length=50
        id_df=X['id'][0]
        sensor_cols = ['s' + str(i) for i in range(1,22)]
        sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
        sequence_cols.extend(sensor_cols)
        seq_gen = (list(gen_sequence(X[X['id']==id], seq_length, sequence_cols)) 
           for id in X['id'].unique())
        seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
        return seq_array


class GenerateLabels(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        seq_length=50
        label_gen = [gen_labels(X[X['id']==id], seq_length, ['RUL']) 
             for id in X['id'].unique()]
        label_array = np.concatenate(label_gen).astype(np.float32)
        label_array.shape
        return label_array

class GenerateLabelsClassification(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        seq_length=50
        label_gen = [gen_labels(X[X['id']==id], seq_length, ['label1']) 
             for id in X['id'].unique()]
        label_array = np.concatenate(label_gen).astype(np.float32)
        label_array.shape
        return label_array
