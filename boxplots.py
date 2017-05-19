import matplotlib.pyplot as plt
import numpy as np

plotAddress = '/home/nahid/UT_research/clueweb12/bpref_result/plots/'
WT2014 = [274, 286, 214, 245, 404, 287, 212, 261, 247, 328, 281, 324, 324, 312, 233, 351, 238, 331, 316, 250, 266, 367, 245, 316, 222, 319, 281, 229, 285, 401, 225, 334, 220, 277, 284, 320, 175, 259, 395, 259, 225, 369, 378, 423, 215, 250, 228, 249, 241, 457]
WT2013 = [322, 231, 384, 323, 354, 420, 232, 249, 185, 336, 423, 435, 201, 305, 367, 387, 326, 341, 513, 473, 368, 175, 265, 351, 487, 404, 246, 157, 160, 172, 171, 292, 391, 298, 185, 175, 161, 169, 249, 396, 285, 202, 342, 174, 145, 202, 421, 194, 223, 207]
gov2 = [317, 581, 1403, 549, 483, 993, 444, 530, 551, 470, 604, 525, 441, 448, 725, 532, 529, 544, 868, 444, 622, 543, 544, 725, 883, 632, 381, 684, 976, 835, 653, 1048, 584, 705, 665, 662, 642, 556, 560, 521, 698, 593, 606, 639, 756, 601, 663, 703, 612, 711]
TREC8 = [2739, 1652, 1046, 1533, 1539, 1420, 1545, 1269, 1476, 1524, 2056, 2113, 1662, 1306, 1309, 1235, 2992, 1748, 1670, 1136, 1763, 2121, 1308, 1747, 1546, 2095, 1528, 1645, 1156, 1709, 1431, 2503, 2162, 1676, 1836, 1949, 1796, 1798, 2015, 1830, 1554, 2679, 2230, 1691, 1404, 2020, 1588, 2408, 1451, 1221]

data = [WT2013, WT2014, gov2, TREC8]
plt.boxplot( data)
plt.xticks([1, 2, 3, 4], ['WT2013', 'WT2014', 'gov2', 'TREC8'])
plt.grid()
plt.xlabel("Data set",size = 16)
plt.ylabel("Number of documents per topic", size = 16)
#plt.show()

plt.tight_layout()
# plt.show()
plt.savefig(plotAddress +'perTopicDocuments.pdf', format='pdf')