else:
    initial_X_train = []
    initial_y_train = []
    train_index_list = []

    # collecting the seed list from the Rankers
    seed_list = Ranker_topic_to_doclist[topic]
    seed_counter = 0
    seed_one_counter = 0
    seed_zero_counter = 0
    ask_for_label = 0


    for index in xrange(0,len(X)):
        if y[index] == 1:
            seed_one_counter = seed_one_counter + 1
            train_index_list.append(index)
            initial_X_train.append(X[index])
            initial_y_train.append(y[index])
            print seed_one_counter
        if seed_one_counter == n_labeled/2 :
            break

    for index in xrange(0,len(X)):

        if y[index] == 0:
            seed_zero_counter = seed_zero_counter + 1
            train_index_list.append(index)
            initial_X_train.append(X[index])
            initial_y_train.append(y[index])
            print seed_zero_counter
        if seed_zero_counter == n_labeled/2:
            break

    if seed_zero_counter != seed_one_counter:
        skipList.append(topic)

    unmodified_train_X = copy.deepcopy(initial_X_train)
    unmodified_train_y = copy.deepcopy(initial_y_train)
    sampling_weight = []

    for sampling_index in xrange(0, len(initial_X_train)):
        sampling_weight.append(1.0)


    if sampling == True:
        if under_sampling == True:
            print "Undersampling in the seed list"
            # rus = RandomUnderSampler(return_indices=True, replacement=True)
            rus = EasyEnsemble(return_indices=True, replacement=True, n_subsets=num_subsets)
            initial_X_train_sampled, initial_y_train_sampled, indices = rus.fit_sample(initial_X_train, initial_y_train)
        else:
            print "Oversampling in the seed list"
            ros = RandomOverSampler()
            initial_X_train_sampled, initial_y_train_sampled = ros.fit_sample(initial_X_train, initial_y_train)
        initial_X_train = initial_X_train_sampled
        initial_y_train = initial_y_train_sampled

        initial_X_train = initial_X_train.tolist()
        initial_y_train = initial_y_train.tolist()

    initial_X_test = []
    initial_y_test = []

    test_index_list = {}
    test_index_counter = 0
    for train_index in xrange(0, len(X)):
        if train_index not in train_index_list:
            initial_X_test.append(X[train_index])
            test_index_list[test_index_counter] = train_index
            test_index_counter = test_index_counter + 1
            initial_y_test.append(y[train_index])

    #initial_X_test = X_train[n_labeled:]
    #initial_y_test = y_train[n_labeled:]

    #initial_X_test = initial_X_test.tolist()
    #initial_y_test = initial_y_test.tolist()

    print "Before Loop Lenght:", len(initial_X_train), len(initial_y_train)
    predictableSize = len(initial_X_test)
    isPredictable = [1] * predictableSize  # initially we will predict all

    loopCounter = 0
    best_model = 0
    learning_batch_size = n_labeled # starts with the seed size

    if train_per_centage_flag == False:
        numberofloop = math.ceil(size / batch_size)
        if numberofloop == 0:
            numberofloop = 1
        print "Number of loop", numberofloop

        while loopCounter<=numberofloop:
            print "Loop:", loopCounter

            loopDocList = []

            if protocol == 'SPL':
                model = LogisticRegression()

            print len(initial_X_train)
            if correction == True:
                model.fit(initial_X_train, initial_y_train, sample_weight=sampling_weight)
            else:
                if under_sampling == True:
                    ens.fit(initial_X_train, initial_y_train)
                else:
                    model.fit(initial_X_train, initial_y_train)

            y_pred_all = {}

            for train_index in train_index_list:
                y_pred_all[train_index] = y[train_index]

            for train_index in xrange(0, len(X)):
                if train_index not in train_index_list:
                    if under_sampling == True:
                        y_pred_all[train_index] = ens.predict(np.array(X[train_index]).reshape(1, -1))
                    else:
                        y_pred_all[train_index] = model.predict(np.array(X[train_index]).reshape(1, -1))[0]

            y_pred = []
            for key, value in y_pred_all.iteritems():
                # print (key,value)
                y_pred.append(value)

            f1score = f1_score(y, y_pred, average='binary')

            if (learning_curve.has_key(learning_batch_size)):
                tmplist = learning_curve.get(learning_batch_size)
                tmplist.append(f1score)
                learning_curve[learning_batch_size] = tmplist
            else:
                tmplist = []
                tmplist.append(f1score)
                learning_curve[learning_batch_size] = tmplist

            learning_batch_size = learning_batch_size + batch_size
            precision = precision_score(y, y_pred, average='binary')
            recall = recall_score(y, y_pred, average='binary')

            print "precision score:", precision
            print "recall score:", recall
            print "f-1 score:", f1score

            if isPredictable.count(1) == 0:
                break

            #if f1score == 1.0:
            #    break
            #    print "BREAKING LOOP BECAUSE F-1 is 1.0"

            # print "f-1 score:", f1score, "precision:", precision, "recall:", recall, "Number of predicted (1): ", np.count_nonzero(y_pred_validation), "Number of predicted (0):", np.prod(y_pred_validation.shape) - np.count_nonzero(y_pred_validation)
            '''
            precision = TP/(TP+FP) as you've just said if predictor doesn't predicts positive class at all - precision is 0.

            recall = TP/(TP+FN), in case if predictor doesn't predict positive class - TP is 0 - recall is 0.

            So now you are dividing 0/0.'''
            # here is queueSize is the number of predictable element
            queueSize = isPredictable.count(1)

            # ===========================================================
            # Protocol CAL
            # ===========================================================
            if protocol == 'CAL':
                CAL()
            # ===========================================================

            # ===========================================================
            # Protocol SAL
            # ===========================================================
            if protocol == 'SAL':
                print "####SAL####"
                queue = Queue.PriorityQueue(queueSize)
                y_prob = []
                counter = 0
                sumForCorrection = 0.0
                for counter in xrange(0,predictableSize):
                    if isPredictable[counter] == 1:
                        # reshapping reshape(1,-1) because it does not take one emelemt array
                        # list does not contain reshape so we are using np,array
                        # model. predit returns two value in index [0] of the list
                        if under_sampling == True:
                            y_prob = ens.predict_proba(np.array(initial_X_test[counter]).reshape(1,-1))
                        else:
                            y_prob = model.predict_proba(np.array(initial_X_test[counter]).reshape(1,-1))[0]
                        entropy = (-1)*(y_prob[0]*log(y_prob[0],2)+y_prob[1]*log(y_prob[1],2))
                        queue.put(relevance(entropy, counter))
                        sumForCorrection = sumForCorrection + entropy


                batch_counter = 0
                while not queue.empty():
                    if batch_counter == batch_size:
                        break
                    item = queue.get()
                    #print len(item)
                    #print item.priority, item.index
                    isPredictable[item.index] = 0 # not predictable
                    if correction == True:
                        correctionWeight = item.priority/sumForCorrection
                        #correctedList = [x / correctionWeight for x in initial_X_test[item.index]]
                        unmodified_train_X.append(initial_X_test[item.index])
                        sampling_weight.append(correctionWeight)
                    else:
                        unmodified_train_X.append(initial_X_test[item.index])
                        sampling_weight.append(1.0)

                    unmodified_train_y.append(initial_y_test[item.index])
                    train_index_list.append(test_index_list[item.index])

                    loopDocList.append(int(initial_y_test[item.index]))
                    batch_counter = batch_counter + 1
                    #print X_train.append(X_test.pop(item.priority))
            # ===========================================================

            # ===========================================================
            # Protocol SPL
            # ===========================================================
            if protocol == 'SPL':
                print "####SPL####"
                randomArray =[]
                randomArrayIndex = 0
                for counter in xrange(0, predictableSize):
                    if isPredictable[counter] == 1:
                        randomArray.append(counter)
                        randomArrayIndex = randomArrayIndex + 1
                import random
                random.shuffle(randomArray)

                batch_counter = 0
                for batch_counter in xrange(0,batch_size):
                    if batch_counter>len(randomArray)-1:
                        break
                    itemIndex = randomArray[batch_counter]
                    isPredictable[itemIndex] = 0
                    unmodified_train_X.append(initial_X_test[itemIndex])
                    unmodified_train_y.append(initial_y_test[itemIndex])
                    sampling_weight.append(1.0)

                    train_index_list.append(test_index_list[itemIndex])
                    loopDocList.append(int(initial_y_test[itemIndex]))
            # ===========================================================

            if iter_sampling == True:
                print "Oversampling in the active iteration list"
                ros = RandomOverSampler()
                initial_X_train = None
                initial_y_train = None
                initial_X_train, initial_y_train = ros.fit_sample(unmodified_train_X, unmodified_train_y)
            else:
                if under_sampling == True:
                    # rus = RandomUnderSampler(return_indices=True, replacement=True)
                    rus = EasyEnsemble(return_indices=True, replacement=True, n_subsets=num_subsets)
                    initial_X_train = None
                    initial_y_train = None
                    initial_X_train, initial_y_train, indices = rus.fit_sample(unmodified_train_X, unmodified_train_y)
                else:
                    initial_X_train[:] = []
                    initial_y_train[:] = []
                    initial_X_train = copy.deepcopy(unmodified_train_X)
                    initial_y_train = copy.deepcopy(unmodified_train_y)

            loopCounter = loopCounter + 1
    else:
        numberofloop = len(train_per_centage)
        train_size_controller = len(initial_X_train)
        while loopCounter < numberofloop:
            size_limit = math.ceil(train_per_centage[loopCounter]*len(X))

            print "Loop:", loopCounter
            print "Initial size:",train_size_controller, "limit:", size_limit

            loopDocList = []

            if protocol == 'SPL':
                model = LogisticRegression()

            print "length of initial X train:", len(initial_X_train)
            print initial_y_train

            if correction == True:
                model.fit(initial_X_train, initial_y_train, sample_weight=sampling_weight)
            else:
                if under_sampling == True:
                    # print(initial_X_train)
                    # print(initial_y_train)
                    ens.fit(initial_X_train, initial_y_train)
                else:
                    model.fit(initial_X_train, initial_y_train)

            y_pred_all = {}

            human_label_str = ""

            for train_index in train_index_list:
                y_pred_all[train_index] = y[train_index]
                docNo = docIndex_DocNo[train_index]
                human_label_str = human_label_str + str(topic) + " " + str(docNo) + " " + str(y_pred_all[train_index]) + "\n"
                #print y_pred_all[train_index]
            human_label_location_final = human_label_location + str(train_per_centage[loopCounter]) + '_human_.txt'
            text_file = open(human_label_location_final, "a")
            text_file.write(human_label_str)
            text_file.close()

            for train_index in xrange(0, len(X)):
                if train_index not in train_index_list:
                    if under_sampling == True:
                        y_pred_all[train_index] = ens.predict(np.array(X[train_index]).reshape(1, -1))
                    else:
                        y_pred_all[train_index] = model.predict(np.array(X[train_index]).reshape(1, -1))[0]

            y_pred = []
            for key, value in y_pred_all.iteritems():
                # print (key,value)
                y_pred.append(value)

            pred_topic_str = ""
            for docIndex in xrange(0, len(X)):
                docNo = docIndex_DocNo[docIndex]
                pred_topic_str = pred_topic_str + str(topic) + " " + str(docNo) + " " + str(y_pred_all[docIndex]) + "\n"
            predicted_location_final = predicted_location_base + str(train_per_centage[loopCounter]) + '.txt'
            text_file = open(predicted_location_final, "a")
            text_file.write(pred_topic_str)
            text_file.close()

            f1score = f1_score(y, y_pred, average='binary')

            if (learning_curve.has_key(train_per_centage[loopCounter])):
                tmplist = learning_curve.get(train_per_centage[loopCounter])
                tmplist.append(f1score)
                learning_curve[train_per_centage[loopCounter]] = tmplist
            else:
                tmplist = []
                tmplist.append(f1score)
                learning_curve[train_per_centage[loopCounter]] = tmplist

            learning_batch_size = learning_batch_size + batch_size
            precision = precision_score(y, y_pred, average='binary')
            recall = recall_score(y, y_pred, average='binary')

            print "precision score:", precision
            print "recall score:", recall
            print "f-1 score:", f1score

            if isPredictable.count(1) == 0:
                break

            # if f1score == 1.0:
            #    break
            #    print "BREAKING LOOP BECAUSE F-1 is 1.0"

            # print "f-1 score:", f1score, "precision:", precision, "recall:", recall, "Number of predicted (1): ", np.count_nonzero(y_pred_validation), "Number of predicted (0):", np.prod(y_pred_validation.shape) - np.count_nonzero(y_pred_validation)
            '''
            precision = TP/(TP+FP) as you've just said if predictor doesn't predicts positive class at all - precision is 0.

            recall = TP/(TP+FN), in case if predictor doesn't predict positive class - TP is 0 - recall is 0.

            So now you are dividing 0/0.'''
            # here is queueSize is the number of predictable element
            queueSize = isPredictable.count(1)

            if protocol == 'CAL':
                print "####CAL####"
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
                        # print y_prob
                        queue.put(relevance(y_prob[1], counter))
                        sumForCorrection = sumForCorrection + y_prob[1]

                batch_counter = 0
                while not queue.empty():
                    if train_size_controller == size_limit:
                        break
                    item = queue.get()
                    # print len(item)
                    # print item.priority, item.index

                    isPredictable[item.index] = 0  # not predictable
                    # initial_X_train.append(initial_X_test[item.index])
                    # initial_y_train.append(initial_y_test[item.index])

                    if correction == True:
                        correctionWeight = item.priority / sumForCorrection
                        # correctedItem = [x / correctionWeight for x in initial_X_test[item.index]]
                        unmodified_train_X.append(initial_X_test[item.index])
                        sampling_weight.append(correctionWeight)
                    else:
                        unmodified_train_X.append(initial_X_test[item.index])
                        sampling_weight.append(1.0)
                    unmodified_train_y.append(initial_y_test[item.index])

                    train_index_list.append(test_index_list[item.index])

                    # print "Docs:", initial_X_test[item.index]
                    loopDocList.append(int(initial_y_test[item.index]))
                    train_size_controller = train_size_controller + 1
                    # print X_train.append(X_test.pop(item.priority))

            if protocol == 'SAL':
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

                batch_counter = 0
                while not queue.empty():
                    if train_size_controller == size_limit:
                        break
                    item = queue.get()
                    # print len(item)
                    # print item.priority, item.index
                    isPredictable[item.index] = 0  # not predictable
                    if correction == True:
                        correctionWeight = item.priority / sumForCorrection
                        # correctedList = [x / correctionWeight for x in initial_X_test[item.index]]
                        unmodified_train_X.append(initial_X_test[item.index])
                        sampling_weight.append(correctionWeight)
                    else:
                        unmodified_train_X.append(initial_X_test[item.index])
                        sampling_weight.append(1.0)

                    unmodified_train_y.append(initial_y_test[item.index])
                    train_index_list.append(test_index_list[item.index])

                    loopDocList.append(int(initial_y_test[item.index]))
                    train_size_controller = train_size_controller + 1
                    # print X_train.append(X_test.pop(item.priority))

            if protocol == 'SPL':
                print "####SPL####"
                randomArray = []
                randomArrayIndex = 0
                for counter in xrange(0, predictableSize):
                    if isPredictable[counter] == 1:
                        randomArray.append(counter)
                        randomArrayIndex = randomArrayIndex + 1
                import random

                random.shuffle(randomArray)

                batch_counter = 0
                for batch_counter in xrange(0, len(randomArray)):
                    #if batch_counter > len(randomArray) - 1:
                    #    break
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

            if iter_sampling == True:
                print "Oversampling in the active iteration list"
                ros = RandomOverSampler()
                initial_X_train = None
                initial_y_train = None
                initial_X_train, initial_y_train = ros.fit_sample(unmodified_train_X, unmodified_train_y)
            else:
                if under_sampling == True:
                    # rus = RandomUnderSampler(return_indices=True, replacement=True)
                    rus = EasyEnsemble(return_indices=True, replacement=True, n_subsets=num_subsets)
                    initial_X_train = None
                    initial_y_train = None
                    initial_X_train, initial_y_train, indices = rus.fit_sample(unmodified_train_X, unmodified_train_y)
                    print(indices)
                    print(initial_y_train)
                else:
                    initial_X_train[:] = []
                    initial_y_train[:] = []
                    initial_X_train = copy.deepcopy(unmodified_train_X)
                    initial_y_train = copy.deepcopy(unmodified_train_y)

            loopCounter = loopCounter + 1


    print "Fininshed loop", len(initial_X_train)
    y_pred_all = {}

    human_label_str = ""

    for train_index in train_index_list:
        y_pred_all[train_index] = y[train_index]
        docNo = docIndex_DocNo[train_index]
        human_label_str = human_label_str + str(topic) + " " + str(docNo) + " " + str(
            y_pred_all[train_index]) + "\n"
    human_label_location_final = human_label_location + str(1.1) + '_human_.txt'
    text_file = open(human_label_location_final, "a")
    text_file.write(human_label_str)
    text_file.close()


    for train_index in xrange(0, len(X)):
        if train_index not in train_index_list:
            if under_sampling == True:
                y_pred_all[train_index] = ens.predict(np.array(X[train_index]).reshape(1, -1))
            else:    
                y_pred_all[train_index] = model.predict(np.array(X[train_index]).reshape(1, -1))[0]

    y_pred = []
    for key, value in y_pred_all.iteritems():
        # print (key,value)
        y_pred.append(value)

    f1score = f1_score(y, y_pred, average='binary')
    precision = precision_score(y, y_pred, average='binary')
    recall = recall_score(y, y_pred, average='binary')


    print "precision score:", precision
    print "recall score:", recall
    print "f-1 score:", f1score

    if train_per_centage_flag == True:
        if (learning_curve.has_key(1.1)):
            tmplist = learning_curve.get(1.1)
            tmplist.append(f1score)
            learning_curve[1.1] = tmplist
        else:
            tmplist = []
            tmplist.append(f1score)
            learning_curve[1.1] = tmplist

        pred_topic_str = ""
        for docIndex in xrange(0, len(X)):
            docNo = docIndex_DocNo[docIndex]
            pred_topic_str = pred_topic_str + str(topic) + " " + str(docNo) + " " + str(
                y_pred_all[docIndex]) + "\n"

        predicted_location_final = predicted_location_base + str(1.1) + '.txt'
        text_file = open(predicted_location_final, "a")
        text_file.write(pred_topic_str)
        text_file.close()

    # print "f-1 score:", f1score, "precision:", precision, "recall:", recall, "Number of predicted (1): ", np.count_nonzero(y_pred_validation), "Number of predicted (0):", np.prod(y_pred_validation.shape) - np.count_nonzero(y_pred_validation)
    '''
    precision = TP/(TP+FP) as you've just said if predictor doesn't predicts positive class at all - precision is 0.

    recall = TP/(TP+FN), in case if predictor doesn't predict positive class - TP is 0 - recall is 0.

    So now you are dividing 0/0.'''


    accuracy = accuracy_score(y, y_pred)
    print "model accuracy (%): ", accuracy * 100, "%"

    s = s + topic + "," + str(datasize) + "," + str(numberOne) + "," + str(numberZero) + "," + str(prevelance) + "," + str(precision) + "," + str(recall) + "," + str(f1score) + "," + str(accuracy) + "," + str(best_f1) +"\n";
    # writing the actual and prediction



    counter = 0
    # doing predcition again for all documents in the validation and test set for writing the prediction files
    
    for docIndex in xrange(0, len(X)):
        docNo = docIndex_DocNo[docIndex]
        pred_str = pred_str + str(topic) + " "+str(docNo) + " " + str(y_pred_all[docIndex]) + "\n"
        counter = counter + 1