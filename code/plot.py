import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import numpy as np 
import os

batch = np.linspace(10, 100, 10)
# folder = 'learning_curve'

# dirs = os.listdir(folder)

# for datasource in dirs:
# 	# print(datasource)
# 	ff = os.path.join(folder, datasource)

# 	for sample_type in os.listdir(ff):
# 		print(datasource, sample_type)



# #=======================================================================
# plt.figure(1)
# # CAL
# f1score = [0.45490030262,0.798710019401,0.908193237154,0.960569115039,0.98430070358,0.991224208488,0.997166144159,0.998383680576,0.999879429469,1.0]
# area = np.trapz(f1score, batch)
# lines = plt.plot(batch, f1score, '-r', label='CAL = ' + str(round(area, 2)))

# # SAL
# f1score = [0.408846930678,0.563256436573,0.67501077819,0.738446605457,0.778322878289,0.80685314516,0.823407856596,0.845465474181,0.881009658406,1.0]
# area = np.trapz(f1score, batch)
# lines = plt.plot(batch, f1score, '-g', label='SAL = ' + str(round(area, 2)))

# # SPL
# f1score = [0.389781434838,0.464278677999,0.500743633432,0.544679318094,0.583411482161,0.62552326294,0.691097073379,0.759515688947,0.861214668146,1.0]
# area = np.trapz(f1score, batch)
# lines = plt.plot(batch, f1score, '-b', label='SPL = ' + str(round(area, 2)))

# plt.title('F1 scores for undersampling + bagging + RDS')
# plt.xlabel("batch size")
# plt.ylabel("F1 score")
# plt.legend(loc='lower right')
# plt.axis([10, 100, 0.4, 1.0])

# #=======================================================================
# plt.figure(2)
# # CAL
# recall = [0.799590517591,0.843526154637,0.911697147018,0.954954785486,0.973467145345,0.987543634398,0.994774319535,0.996798737317,0.999759584146,1.0]
# area = np.trapz(recall, batch)
# lines = plt.plot(batch, recall, '-r', label='CAL = ' + str(round(area, 2)))

# # SAL
# recall = [0.852876755727,0.909428311499,0.946024902729,0.962884310257,0.97504224854,0.984128179153,0.992787728908,0.997358281764,0.999636127355,1.0]
# area = np.trapz(recall, batch)
# lines = plt.plot(batch, recall, '-g', label='SAL = ' + str(round(area, 2)))

# # SPL
# recall = [0.848725789721,0.866104866927,0.894215741358,0.915115405929,0.927466721107,0.946835028301,0.95797999961,0.97205466827,0.985908659865,1.0]
# area = np.trapz(recall, batch)
# lines = plt.plot(batch, recall, '-b', label='SPL = ' + str(round(area, 2)))

# plt.title('Recall for undersampling + bagging + RDS')
# plt.xlabel("batch size")
# plt.ylabel("recall score")
# plt.legend(loc='lower right')
# plt.axis([10, 100, 0.4, 1.0])

# #=======================================================================
# plt.figure(3)

# # CAL
# specificity = [0.899324231579,0.980871432789,0.996180654626,0.999106005369,0.999862358823,0.999955758848,0.9999864682,1.0,1.0,1.0]
# area = np.trapz(specificity, batch)
# lines = plt.plot(batch, specificity, '-r', label='CAL = ' + str(round(area, 2)))

# # SAL
# specificity = [0.858791630291,0.914756024079,0.942900441256,0.959119031447,0.970066982529,0.977701724514,0.982103156607,0.985920632937,0.990148425702,1.0]
# area = np.trapz(specificity, batch)
# lines = plt.plot(batch, specificity, '-g', label='SAL = ' + str(round(area, 2)))

# # SPL
# specificity = [0.854818605257,0.904422256877,0.918794293695,0.931180769471,0.942810903175,0.952874108057,0.966487293602,0.976700616571,0.988925791464,1.0]
# area = np.trapz(specificity, batch)
# lines = plt.plot(batch, specificity, '-b', label='SPL = ' + str(round(area, 2)))

# plt.title('Specificity for undersampling + bagging + RDS')
# plt.xlabel("batch size")
# plt.ylabel("specificity score")
# plt.legend(loc='lower right')
# plt.axis([10, 100, 0.4, 1.0])

# #=======================================================================

# plt.figure(4)
# # CAL
# f1score = [0.441272392137,0.707090848233,0.89819137056,0.966844150656,0.980915594633,0.9922421788,0.995809504396,0.998669767246,0.998804132687,1.0]
# area = np.trapz(f1score, batch)
# lines = plt.plot(batch, f1score, '-r', label='CAL = ' + str(round(area, 2)))

# # SAL
# f1score = [0.274902824725,0.352715664158,0.419205031194,0.484149003043,0.535823144567,0.582895722603,0.632864777279,0.681376832274,0.749300201969,1.0]
# area = np.trapz(f1score, batch)
# lines = plt.plot(batch, f1score, '-g', label='SAL = ' + str(round(area, 2)))

# # SPL
# f1score = [0.306074085233,0.363344408351,0.414477665728,0.460075586833,0.521823108767,0.574719236251,0.645524168743,0.730335085456,0.844448651755,1.0]
# area = np.trapz(f1score, batch)
# lines = plt.plot(batch, f1score, '-b', label='SPL = ' + str(round(area, 2)))

# plt.title('F1 scores for undersampling + bagging + IS')
# plt.xlabel("batch size")
# plt.ylabel("F1 score")
# plt.legend(loc='lower right')
# plt.axis([10, 100, 0.4, 1.0])

# #=======================================================================
# plt.figure(5)
# # CAL
# recall = [0.761640752484,0.822792619258,0.907989350383,0.955881090202,0.976199240466,0.987805293402,0.992230170225,0.997385938577,0.999353243088,1.0]
# area = np.trapz(recall, batch)
# lines = plt.plot(batch, recall, '-r', label='CAL = ' + str(round(area, 2)))

# # SAL
# recall = [0.87651247047,0.931760775486,0.962884331352,0.978213258823,0.988095140144,0.994631516563,0.996422875362,0.998165004748,0.99987654321,1.0]
# area = np.trapz(recall, batch)
# lines = plt.plot(batch, recall, '-g', label='SAL = ' + str(round(area, 2)))

# # SPL
# recall = [0.850373298919,0.877663591469,0.884693934413,0.912758601151,0.928235573842,0.946578671333,0.955219276279,0.967584642238,0.988089612796,1.0]
# area = np.trapz(recall, batch)
# lines = plt.plot(batch, recall, '-b', label='SPL = ' + str(round(area, 2)))

# plt.title('Recall for undersampling + bagging + IS')
# plt.xlabel("batch size")
# plt.ylabel("recall score")
# plt.legend(loc='lower right')
# plt.axis([10, 100, 0.4, 1.0])

# #=======================================================================
# plt.figure(6)

# # CAL
# specificity = [0.847316114297,0.966307696264,0.995480974043,0.999288853121,0.999725073359,0.999965069648,0.999976190476,1.0,0.999982532751,1.0]
# area = np.trapz(specificity, batch)
# lines = plt.plot(batch, specificity, '-r', label='CAL = ' + str(round(area, 2)))

# # SAL
# specificity = [0.702839667868,0.776766185476,0.82082481689,0.862887822319,0.892890094894,0.915167085032,0.937256542081,0.955568277884,0.975268150648,1.0]
# area = np.trapz(specificity, batch)
# lines = plt.plot(batch, specificity, '-g', label='SAL = ' + str(round(area, 2)))

# # SPL
# specificity = [0.802832144619,0.849922645383,0.889317105081,0.907648646737,0.929147359918,0.943483468288,0.959399565881,0.973599214449,0.987190132625,1.0]
# area = np.trapz(specificity, batch)
# lines = plt.plot(batch, specificity, '-b', label='SPL = ' + str(round(area, 2)))

# plt.title('Specificity for undersampling + bagging + IS')
# plt.xlabel("batch size")
# plt.ylabel("specificity score")
# plt.legend(loc='lower right')

#=======================================================================
plt.figure(1)

# clef
F1 = [0.975939771645,0.996346197422,0.999345916523,0.999821907074,0.99997436714,0.9999864682,1.0,1.0,1.0,1.0]
area = np.trapz(F1, batch)
lines = plt.plot(batch, F1, '-r', label='CAL specificity oversampling = ' + str(round(area, 2)))

# # gov2
F1 = [0.847316114297,0.966307696264,0.995480974043,0.999288853121,0.999725073359,0.999965069648,0.999976190476,1.0,0.999982532751,1.0]
area = np.trapz(F1, batch)
lines = plt.plot(batch, F1, '-g', label='CAL specificity undersampling = ' + str(round(area, 2)))

# # WT13
# F1 = [0.548741296859,0.714657405445,0.830602180225,0.898973220842,0.933583198074,0.961420385091,0.982542654933,0.992506919151,0.997962254214,1.0]
# area = np.trapz(F1, batch)
# lines = plt.plot(batch, F1, '-b', label='none = ' + str(round(area, 2)))

# # WT14
# F1 = [0.588365471349,0.656030703746,0.719997614091,0.781607307729,0.85340653802,0.902980206423,0.945129475867,0.971095956123,0.991329346247,1.0]
# area = np.trapz(F1, batch)
# lines = plt.plot(batch, F1, '-y', label='WT14 = ' + str(round(area, 2)))

# plt.title('F1-score for CLEF + bagging + IS + CAL')
plt.xlabel("batch size")
plt.ylabel("F1 score")
plt.legend(loc='lower right')

plt.axis([10, 100, 0.4, 1.0])
# plt.show()
plt.savefig('sesuatu.jpg')