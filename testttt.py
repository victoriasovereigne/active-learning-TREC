import matplotlib.pyplot as plt
x_labels_set =[0.1, 0.2]

protocol_result = [0.73753435729546446, 0.73534257614451215]
plt.plot(x_labels_set, protocol_result, '-r', label='SAL', linewidth=2.0)
#plt.plot(x_labels_set, protocol_result['CAL'], '-b', label='CAL', linewidth=2.0)
#plt.plot(x_labels_set, protocol_result['SPL'], '-g', label='SPL', linewidth=2.0)

plt.xlabel('Percentage of human judgements')

plt.ylabel('tau correlation')
plt.ylim([0.0, 1])
plt.legend(loc=2)
plt.grid()
plt.tight_layout()
plt.show()
