import matplotlib.pyplot as plt


def lst_to_str(lst):
    s = ""
    for i in lst:
        s += str(i)
        s += " "
    return s


class resultGenerator(object):
    def __init__(self):
        self.modeldata = []
        self.auc = []

    def plot(self):
        for i in range(len(self.modeldata)):
            plt.bar(self.modeldata[i], self.auc[i])

        plt.title("AUC Score VS Model + Dataset")

        plt.xlabel("Model + Dataset")

        plt.ylabel("AUC Score")
        # plt.show()

        plt.savefig('./img/evaluationResult.jpg')
        re = open('./img/result.txt', mode='w')
        re.write(lst_to_str(self.modeldata))
        re.write('\n')
        re.write(lst_to_str(self.auc))
        re.write('\n')
        re.close()

        plt.close()

    def plot_line(self):
        plt.title("AUC Score VS Model + Dataset")

        plt.xlabel("Model + Dataset")

        plt.ylabel("AUC Score")
        y_ticks = range(5)
        plt.yticks(y_ticks[::1])
        plt.plot(self.modeldata, self.auc, linewidth=1, color='blue', marker='o', markerfacecolor='blue', markersize=4)
        # plt.plot(x1,y1,label='firt line',linewidth=1,color='blue',marker='o', markerfacecolor='blue',markersize=4)
        plt.savefig('./img/evaluationResult2.jpg')

