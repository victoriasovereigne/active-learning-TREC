from helper import *
import Queue
import numpy as np
import undersampler
from math import log
import random

def empty_queue(queue, mycounter, limit, isPredictable, correction, unmodified_train_X, unmodified_train_y, 
    initial_X_test, initial_y_test, sampling_weight, train_index_list, test_index_list, loopDocList):
    while not queue.empty():
        if mycounter == limit:
            break
        item = queue.get()
        isPredictable[item.index] = 0  # not predictable

        if correction == True:
            correctionWeight = item.priority / sumForCorrection
            unmodified_train_X.append(initial_X_test[item.index])
            sampling_weight.append(correctionWeight)
        else:
            unmodified_train_X.append(initial_X_test[item.index])
            sampling_weight.append(1.0)

        unmodified_train_y.append(initial_y_test[item.index])
        train_index_list.append(test_index_list[item.index])
        loopDocList.append(int(initial_y_test[item.index]))
        mycounter = mycounter + 1

    return mycounter

def CAL(queueSize, predictableSize, isPredictable, under_sampling, ens, model, initial_X_test):
    print "####CAL#### this method works"
    queue = Queue.PriorityQueue(queueSize)
    y_prob = []
    counter = 0
    sumForCorrection = 0.0
    for counter in xrange(0, predictableSize):
        if isPredictable[counter] == 1:
            # reshapping reshape(1,-1) because it does not take one emelemt array
            # list does not contain reshape so we are using np,array
            # model. predit returns two value in index [0] of the list
            if under_sampling == True:
                y_prob = ens.predict_proba(np.array(initial_X_test[counter]).reshape(1, -1))
            else:
                y_prob = model.predict_proba(np.array(initial_X_test[counter]).reshape(1, -1))[0]
            
            queue.put(relevance(y_prob[1], counter))
            sumForCorrection = sumForCorrection + y_prob[1]

    return queue


def SAL(queueSize, predictableSize, isPredictable, under_sampling, ens, model, initial_X_test):
    print "####SAL####"
    queue = Queue.PriorityQueue(queueSize)
    y_prob = []
    counter = 0
    sumForCorrection = 0.0
    for counter in xrange(0, predictableSize):
        if isPredictable[counter] == 1:
            # reshapping reshape(1,-1) because it does not take one emelemt array
            # list does not contain reshape so we are using np,array
            # model. predit returns two value in index [0] of the list
            if under_sampling == True:
                y_prob = ens.predict_proba(np.array(initial_X_test[counter]).reshape(1, -1))
            else:
                y_prob = model.predict_proba(np.array(initial_X_test[counter]).reshape(1, -1))[0]
            entropy = (-1) * (y_prob[0] * log(y_prob[0], 2) + y_prob[1] * log(y_prob[1], 2))
            queue.put(relevance(entropy, counter))
            sumForCorrection = sumForCorrection + entropy

    return queue

def SPL_shuffle_array(predictableSize, isPredictable):
    print "####SPL####"
    randomArray = []
    randomArrayIndex = 0
    for counter in xrange(0, predictableSize):
        if isPredictable[counter] == 1:
            randomArray.append(counter)
            randomArrayIndex = randomArrayIndex + 1
    
    random.shuffle(randomArray)
    return randomArray

def SPL_train_percentage_false(batch_size, randomArray, isPredictable, 
    unmodified_train_X, unmodified_train_y, initial_X_test, initial_y_test,
    sampling_weight, train_index_list, test_index_list, loopDocList):
    batch_counter = 0
    for batch_counter in xrange(0, batch_size):
        if batch_counter > len(randomArray) - 1:
            break
        itemIndex = randomArray[batch_counter]
        isPredictable[itemIndex] = 0
        unmodified_train_X.append(initial_X_test[itemIndex])
        unmodified_train_y.append(initial_y_test[itemIndex])
        sampling_weight.append(1.0)
        train_index_list.append(test_index_list[itemIndex])
        loopDocList.append(int(initial_y_test[itemIndex]))

def SPL_train_percentage_true(train_size_controller, size_limit, randomArray, isPredictable, 
    unmodified_train_X, unmodified_train_y, initial_X_test, initial_y_test,
    sampling_weight, train_index_list, test_index_list, loopDocList):
    batch_counter = 0
    for batch_counter in xrange(0, len(randomArray)):
        if train_size_controller == size_limit:
            break
        itemIndex = randomArray[batch_counter]
        isPredictable[itemIndex] = 0
        unmodified_train_X.append(initial_X_test[itemIndex])
        unmodified_train_y.append(initial_y_test[itemIndex])
        sampling_weight.append(1.0)

        train_index_list.append(test_index_list[itemIndex])
        loopDocList.append(int(initial_y_test[itemIndex]))
        train_size_controller = train_size_controller + 1

    return train_size_controller
