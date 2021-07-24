import pdb
import numpy as np
import code_for_hw3_part2 as hw3

#-------------------------------------------------------------------------------
# Auto Data
#-------------------------------------------------------------------------------

# Returns a list of dictionaries.  Keys are the column names, including mpg.
auto_data_all = hw3.load_auto_data('auto-mpg.tsv')

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are hw3.standard and hw3.one_hot.
# 'name' is not numeric and would need a different encoding.
features = [('cylinders', hw3.raw),
            ('displacement', hw3.raw),
            ('horsepower', hw3.raw),
            ('weight', hw3.raw),
            ('acceleration', hw3.raw),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.raw)]

features2 = [('cylinders', hw3.one_hot),
            ('displacement', hw3.standard),
            ('horsepower', hw3.standard),
            ('weight', hw3.standard),
            ('acceleration', hw3.standard),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.one_hot)]


# Construct the standard data and label arrays
auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_all, features2)
print('auto data and labels shape', auto_data.shape, auto_labels.shape)

if False:                               # set to True to see histograms
    import matplotlib.pyplot as plt
    for feat in range(auto_data.shape[0]):
        print('Feature', feat, features[feat][0])
        # Plot histograms in one window, different colors
        plt.hist(auto_data[feat,auto_labels[0,:] > 0])
        plt.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()
        # Plot histograms in two windows, different colors
        fig,(a1,a2) = plt.subplots(nrows=2)
        a1.hist(auto_data[feat,auto_labels[0,:] > 0])
        a2.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()

#-------------------------------------------------------------------------------
# Analyze auto data
#-------------------------------------------------------------------------------

#print(hw3.xval_learning_alg(hw3.perceptron, auto_data, auto_labels, 10))
#print(hw3.xval_learning_alg(hw3.averaged_perceptron, auto_data, auto_labels, 10))
#learned = hw3.averaged_perceptron(auto_data, auto_labels)
#print(learned)

#-------------------------------------------------------------------------------
# Review Data
#-------------------------------------------------------------------------------

# Returns lists of dictionaries.  Keys are the column names, 'sentiment' and 'text'.
# The train data has 10,000 examples
review_data = hw3.load_review_data('reviews.tsv')

# Lists texts of reviews and list of labels (1 or -1)
review_texts, review_label_list = zip(*((sample['text'], sample['sentiment']) for sample in review_data))

# The dictionary of all the words for "bag of words"
dictionary = hw3.bag_of_words(review_texts)

# The standard data arrays for the bag of words
review_bow_data = hw3.extract_bow_feature_vectors(review_texts, dictionary)
review_labels = hw3.rv(review_label_list)
print('review_bow_data and labels shape', review_bow_data.shape, review_labels.shape)

#-------------------------------------------------------------------------------
# Analyze review data
#-------------------------------------------------------------------------------
"""
params = hw3.averaged_perceptron(review_bow_data, review_labels, params={'T':10})
top = np.argsort(params[0], axis=0)
first10 = top[top.shape[0]-10:].flatten()
last10 = top[:10].flatten()
rd = hw3.reverse_dict(dictionary)
pos = []
neg = []
print("Top positive:")
for i in reversed(first10):
    pos.append(rd[i])
print(pos)
print("Top negative: ")
for i in reversed(last10):
    neg.append(rd[i])
print(neg)
"""

#-------------------------------------------------------------------------------
# MNIST Data
#-------------------------------------------------------------------------------

"""
Returns a dictionary formatted as follows:
{
    0: {
        "images": [(m by n image), (m by n image), ...],
        "labels": [0, 0, ..., 0]
    },
    1: {...},
    ...
    9
}
Where labels range from 0 to 9 and (m, n) images are represented
by arrays of floats from 0 to 1
"""
mnist_data_all = hw3.load_mnist_data(range(10))

print('mnist_data_all loaded. shape of single images is', mnist_data_all[0]["images"][0].shape)

# HINT: change the [0] and [1] if you want to access different images
d0 = mnist_data_all[6]["images"]
d1 = mnist_data_all[8]["images"]
y0 = np.repeat(-1, len(d0)).reshape(1,-1)
y1 = np.repeat(1, len(d1)).reshape(1,-1)

# data goes into the feature computation functions
data = np.vstack((d0, d1))
# labels can directly go into the perceptron algorithm
labels = np.vstack((y0.T, y1.T)).T

def raw_mnist_features(x):
    return np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2])).T

def row_average_features(x):
    """
    @param x (m,n) array with values in (0,1)
    @return (m,1) array where each entry is the average of a row
    """
    m = x.shape[0]
    res = np.mean(x, axis=1)
    res = np.reshape(res, (m, 1))
    return res



def col_average_features(x):
    """
    @param x (m,n) array with values in (0,1)
    @return (n,1) array where each entry is the average of a column
    """
    n = x.shape[1]
    res = np.mean(x, axis=0)
    res = np.reshape(res, (n, 1))
    return res


def top_bottom_features(x):
    """
    @param x (m,n) array with values in (0,1)
    @return (2,1) array where the first entry is the average of the
    top half of the image = rows 0 to floor(m/2) [exclusive]
    and the second entry is the average of the bottom half of the image
    = rows floor(m/2) [inclusive] to m
    """
    res = np.mean(x, axis=1)
    top = np.mean(res[0:int(res.shape[0]/2)])
    bot = np.mean(res[int(res.shape[0]/2):res.shape[0]])
    return np.array([top, bot]).T

# use this function to evaluate accuracy
#acc = hw3.get_classification_accuracy(raw_mnist_features(data), labels)


#-------------------------------------------------------------------------------
# Analyze MNIST data
#-------------------------------------------------------------------------------

#data_flat = raw_mnist_features(data)
#acc = hw3.get_classification_accuracy(data_flat, labels)
#print(acc)

rowa = np.zeros((data.shape[0], data.shape[1]))
cola = np.zeros((data.shape[0], data.shape[1]))
topbot = np.zeros((data.shape[0], 2))
for m in range(data.shape[0]):
    rowa[m] = row_average_features(data[m]).flatten()
    cola[m] = col_average_features(data[m]).flatten()
    topbot[m] = top_bottom_features(data[m]).flatten()
acc1 = hw3.get_classification_accuracy(rowa.T, labels)
acc2 = hw3.get_classification_accuracy(cola.T, labels)
acc3 = hw3.get_classification_accuracy(topbot.T, labels)
print([acc1, acc2, acc3])
