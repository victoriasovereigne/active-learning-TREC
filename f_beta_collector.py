from sklearn.metrics import fbeta_score
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(5, 2)

os.chdir('/home/nahid/Downloads/trec_eval.9.0/')

base_address1 = "/home/nahid/UT_research/clueweb12/complete_result/"
plotAddress = "/home/nahid/UT_research/clueweb12/complete_result/plots/varyf1/"
baseAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/"

#protocol_list = ['SAL','CAL', 'SPL']
protocol_list = ['CAL']
#dataset_list = ['WT2013']
dataset_list = ['WT2013', 'WT2014', 'gov2', 'TREC8']
ranker_list = ['True', 'False']
sampling_list = ['True','False']
train_per_centage_flag = 'True'
seed_size =  [10] #50      # number of samples that are initially labeled
batch_size = [25] #50
train_per_centage = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1] # skiping seed part which is named as 0.1
#train_per_centage = [0.2, 0.3] # skiping seed part which is named as 0.1
x_labels_set_name = ['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%']
#x_labels_set =[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
x_labels_set =[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#x_labels_set =[10,20]


result_location = ''
counter = 0
missing = 0
list = []
protocol_result = {}
#subplot_loc = [521, 522, 523, 524,525, 526, 527, 528, 529]
#subplot_loc = [331, 332, 333, 334,335, 336, 337, 338, 339]
subplot_loc = [221,222,223,224]
start_topic = 0
end_topic = 0
topicSkipList = [202,210,225,234,235,238,244,251,255,262,269,271,278,283,289,291,803,805]

#f_beta_0.5 =



for use_ranker in ranker_list:
    for iter_sampling in sampling_list:
        fig, ax = plt.subplots(nrows=2, ncols=2)
        var = 0
        s = ""
        s1 = ""
        originalqrelMap = []
        predictedqrelMap = []
        for datasource in dataset_list:  # 1
            originalqrelMap = []
            predictedqrelMap = []
            if datasource == 'gov2':
                originAdress = "/media/nahid/Windows8_OS/unzippedsystemRanking/" + datasource + "/"
                #qrelAdress = '/media/nahid/Windows8_OS/finalDownlaod/TREC/gov2/qrels.tb06.top50.txt'
                qrelAdress = '/media/nahid/Windows8_OS/finalDownlaod/TREC/gov2/modified_qreldocsgov2.txt'
                destinationBase = "/media/nahid/Windows8_OS/modifiedSystemRanking/" + datasource + "/"
                predictionAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/gov2/prediction/"
                predictionModifiedAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/gov2/modifiedprediction/"
                start_topic = 801
                end_topic = 851
            elif datasource == 'TREC8':
                originAdress = "/media/nahid/Windows8_OS/unzippedsystemRanking/" + datasource + "/"
                qrelAdress = '/media/nahid/Windows8_OS/finalDownlaod/TREC/TREC8/relevance.txt'
                destinationBase = "/media/nahid/Windows8_OS/modifiedSystemRanking/" + datasource + "/"
                predictionAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/TREC8/prediction/"
                predictionModifiedAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/TREC8/modifiedprediction/"
                start_topic = 401
                end_topic = 451
            elif datasource == 'WT2013':
                originAdress = "/media/nahid/Windows8_OS/unzippedsystemRanking/" + datasource + "/"
                qrelAdress = '/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2013/modified_qreldocs2013.txt'
                destinationBase = "/media/nahid/Windows8_OS/modifiedSystemRanking/" + datasource + "/"
                predictionAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2013/prediction/"
                predictionModifiedAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2013/modifiedprediction/"
                start_topic = 201
                end_topic = 251

            else:
                originAdress = "/media/nahid/Windows8_OS/unzippedsystemRanking/" + datasource + "/"
                qrelAdress = '/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2014/modified_qreldocs2014.txt'
                destinationBase = "/media/nahid/Windows8_OS/modifiedSystemRanking/" + datasource + "/"
                predictionAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2014/prediction/"
                predictionModifiedAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2014/modifiedprediction/"
                start_topic = 251
                end_topic = 301

            print "Original Part"

            originalLabel = {}
            file = open(qrelAdress)
            for lines in file:
                values = lines.split()
                topicNo = values[0]
                docNo = values[2]
                label = int(values[3])
                if label > 1:
                    label = 1
                if label < 0:
                    label = 0
                key = topicNo +":"+docNo
                originalLabel[key] = label
            file.close()



            base_address2 = base_address1 + str(datasource) + "/"
            if use_ranker == 'True':
                base_address3 = base_address2 + "ranker/"
                s1 = "Ranker and "
            else:
                base_address3 = base_address2 + "no_ranker/"
                s1 = "Interactive Search and "
            if iter_sampling == 'True':
                base_address4 = base_address3 + "oversample/"
                s1 = s1 + "oversampling"
            else:
                base_address4 = base_address3 + "htcorrection/"
                s1 = s1 + "HT correction"

            training_variation = []
            for seed in seed_size:  # 2
                for batch in batch_size:  # 3
                    for protocol in protocol_list:  # 4
                        print "Dataset", datasource, "Protocol", protocol, "Seed", seed, "Batch", batch
                        s = "Dataset:" + str(datasource) + ", Seed:" + str(seed) + ", Batch:" + str(batch)
                        list = []
                        for fold in xrange(1, 2):
                            predicted_location_base = base_address4 + 'prediction_protocol:' + protocol + '_batch:' + str(
                                batch) + '_seed:' + str(seed) + '_fold' + str(fold) + '_'
                            f_beta_a = [] # store average value per percentage for beta = 0.25
                            f_beta_b = []  # store average value per percentage for beta = 0.5
                            f_beta_c = []  # store average value per percentage for beta = 1
                            f_beta_d = []  # store average value per percentage for beta = 3
                            f_beta_e = []  # store average value per percentage for beta = 5

                            for percentage in train_per_centage:
                                predictionqrel = predicted_location_base + str(percentage) + '.txt'
                                fbeta_point_twofive = []
                                fbeta_point_five = []
                                fbeta_one = []
                                fbeta_three = []
                                fbeta_five = []
                                for topic in xrange(start_topic, end_topic):
                                    #print "Topic:", topic
                                    if topic in topicSkipList:
                                        #print "Skipping Topic :", topic
                                        continue
                                    topic = str(topic)

                                    f = open(predictionqrel)
                                    s=""

                                    y_original = []
                                    y_pred = []


                                    for lines in f:
                                        values = lines.split()
                                        topicNo = values[0]
                                        if topicNo != topic:
                                            continue
                                        docNo = values[1]
                                        label = int(values[2])
                                        keys = topicNo +":"+docNo
                                        if keys in originalLabel:
                                            y_original.append(originalLabel[keys])
                                            y_pred.append(label)

                                    fbeta_point_twofive.append(fbeta_score(y_original, y_pred, average='binary', beta=0.25))
                                    fbeta_point_five.append(fbeta_score(y_original, y_pred, average='binary', beta=0.50))
                                    fbeta_one.append(fbeta_score(y_original, y_pred, average='binary', beta=1.0))
                                    fbeta_three.append(fbeta_score(y_original, y_pred, average='binary', beta=3.0))
                                    fbeta_five.append(fbeta_score(y_original, y_pred, average='binary', beta=5.0))

                                    f.close()
                                f_beta_a.append(sum(fbeta_point_twofive)/float(len(fbeta_point_twofive)))
                                f_beta_b.append(sum(fbeta_point_five) / float(len(fbeta_point_five)))
                                f_beta_c.append(sum(fbeta_one) / float(len(fbeta_one)))
                                f_beta_d.append(sum(fbeta_three) / float(len(fbeta_three)))
                                f_beta_e.append(sum(fbeta_five) / float(len(fbeta_five)))



            print len(f_beta_a), len(x_labels_set)
            plt.subplot(subplot_loc[var])
            '''plt.plot(x_labels_set, protocol_result['SAL'], '-r', label='SAL', linewidth=2.0)
            #print protocol_result['SAL']
            plt.plot(x_labels_set, protocol_result['CAL'], '-b', label='CAL', linewidth=2.0)
            plt.plot(x_labels_set, protocol_result['SPL'], '-g', label='SPL', linewidth=2.0)
            '''

            plt.plot(x_labels_set, f_beta_a,  marker='o', label='beta = 0.25', linewidth=1.0)
            plt.plot(x_labels_set, f_beta_b,  marker='^', label='beta = 0.50', linewidth=1.0)
            plt.plot(x_labels_set, f_beta_c,  marker='s', label='beta = 1.0', linewidth=1.0)
            plt.plot(x_labels_set, f_beta_d,  marker='v', label='beta = 3.0', linewidth=1.0)
            plt.plot(x_labels_set, f_beta_e,  marker='p', label='beta = 5.0', linewidth=1.0)

            plt.xlabel('Percentage of human judgements',size = 8)

            plt.ylabel('F-beta measure',size = 8)
            plt.ylim([0.5, 1])
            plt.legend(loc=4, fontsize = 6)
            plt.title(datasource,size = 8)
            plt.grid()
            var = var + 1


        plt.suptitle(s1, size=8)
        plt.tight_layout()

        plt.savefig(plotAddress + s1 + '.pdf', format='pdf')



'''fig = plt.figure(figsize=(17,5))
percentile_list = pd.DataFrame(
    {'original': originalqrelMap,
     'predicted': predictedqrelMap
    })
seaborn.regplot(x='original', y='predicted', fit_reg=True, data=percentile_list)
fig.tight_layout()
plt.show()


tau, p_value = kendalltau(originalqrelMap,predictedqrelMap)
print "Kendall's Tau", tau, "and p_value:", p_value

tau, p_value = pearsonr(originalqrelMap,predictedqrelMap)
print "Pearson's r", tau, "and p_value:", p_value
'''