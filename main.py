from DKTP_MODULE.dktp import DKTP
from DKVMN_MODULE.dkvmn import DKVMN
from KQN_MODULE.kqn import KQN
from KTStrategy import KTStrategy
from DKT_MODULE.dkt import DKT

from SAINT_MODULE.saint import SAINT
from SAKT_MODULE.sakt import SAKT
from resultGenerator import resultGenerator

from DKT_MODULE.dkt_p1_e import DKT_P1_E
from DKT_MODULE.dkt_p2_e import DKT_P2_E
from DKVMN_MODULE.dkvmn_p1 import DKVMN_P1
from DKVMN_MODULE.dkvmn_p2 import DKVMN_P2
from SAKT_MODULE.sakt_p1 import SAKT_P1
from SAKT_MODULE.sakt_p2 import SAKT_P2
from SAKT_MODULE.sakt_p3 import SAKT_P3
from SAKT_MODULE.sakt_p3i import SAKT_P3I
from SAKT_MODULE.saktm import SAKTM
from SAINT_MODULE.mysaint import MYSAINT

MODEL_NAME = ['DKT', 'DKVMN', 'DKTP', 'KQN', 'GKT', 'SAKT', 'SAINT']
MODEL_NAME = ['DKT']
MODEL_NAME = ['DKT', 'DKVMN', 'DKTP', 'KQN', 'SAKT', 'SAINT']
# Store information for datasets
# DATASET_NAME = ['ASS09', 'ASS12', 'ASS15', 'ASSCH', 'STATICS', 'KDD05']
# DATASET_NAME = ['ASS09', 'ASS12', 'ASS15', 'ASSCH', 'STATICS', 'JUNYI', 'KDD05', 'KDD06', 'KDDBG']
# DATASET_NAME = ['ASS09', 'ASS15', 'ASSCH', 'STATICS', 'KDD05', 'KDD06', 'KDDBG']
# DATASET_NAME = ['ASSCH', 'STATICS', 'KDD05']
# DATASET_NAME = ['ASS09', 'ASS15', 'STATICS']
# DATASET_NAME = ['ass09', 'ass12', 'ass15', 'assch', 'stat', 'junyi10', 'junyi18', 'alg05', 'alg06', 'algbg', 'poj']

# rich feature one
DATASET_NAME = ['ass09', 'ass12', 'assch', 'stat', 'junyi10', 'alg05', 'alg06', 'algbg']
DATASET_NAME = ['ass09', 'ass12', 'ass15', 'assch', 'stat', 'junyi10', 'alg05', 'alg06', 'algbg', 'poj']

# Choose the KT model as required
class Context:
    def __init__(self):
        self.KTStrategy = None

    def set_model(self, KTStrategy):
        self.KTStrategy = KTStrategy()

    def show_model(self):
        self.KTStrategy.show_model()


# The first strategy: test 1 model with 1 dataset

def strategy1():
    print("test 1 model with 1 dataset")
    ctx = Context()
    ctx.set_model(KTStrategy)
    print("type the model you prefer,", MODEL_NAME)
    model_name = input()
    # model_name = 'MS'
    # model_name = 'DKT'
    # model_name = 'DKVMN'
    # model_name = 'DKTP'
    print("type the dataset you prefer,", DATASET_NAME)
    data_name = input()
    # data_name = 'ass09'
    # file_name = 'ASS09'

    if model_name == 'DKT':
        ctx.set_model(DKT)
        ctx.show_model()
        ctx.KTStrategy.set_data_name(data_name)
        print(ctx.KTStrategy.data_name)
        ctx.KTStrategy.run_model()

    elif model_name == 'MS':
        ctx.set_model(MYSAINT)
        ctx.show_model()
        ctx.KTStrategy.set_data_name(data_name)
        print(ctx.KTStrategy.data_name)
        ctx.KTStrategy.run_model()

    elif model_name == 'DKT1e':
        ctx.set_model(DKT_P1_E)
        ctx.show_model()
        ctx.KTStrategy.set_data_name(data_name)
        print(ctx.KTStrategy.data_name)
        ctx.KTStrategy.run_model()
    elif model_name == 'DKVMN1e':
        ctx.set_model(DKVMN_P1)
        ctx.show_model()
        ctx.KTStrategy.set_data_name(data_name)
        print(ctx.KTStrategy.data_name)
        ctx.KTStrategy.run_model()
    elif model_name == 'DKVMN2e':
        ctx.set_model(DKVMN_P2)
        ctx.show_model()
        ctx.KTStrategy.set_data_name(data_name)
        print(ctx.KTStrategy.data_name)
        ctx.KTStrategy.run_model()
    elif model_name == 'DKT2e':
        ctx.set_model(DKT_P2_E)
        ctx.show_model()
        ctx.KTStrategy.set_data_name(data_name)
        print(ctx.KTStrategy.data_name)
        ctx.KTStrategy.run_model()

    elif model_name == 'DKVMN':
        ctx.set_model(DKVMN)
        ctx.show_model()
        ctx.KTStrategy.set_data_name(data_name)
        print(ctx.KTStrategy.data_name)
        ctx.KTStrategy.run_model()

    elif model_name == 'DKTP':

        ctx.set_model(DKTP)
        ctx.show_model()
        ctx.KTStrategy.set_data_name(data_name)
        print(ctx.KTStrategy.data_name)
        ctx.KTStrategy.run_model()

    elif model_name == 'KQN':

        ctx.set_model(KQN)
        ctx.show_model()
        ctx.KTStrategy.set_data_name(data_name)
        print(ctx.KTStrategy.data_name)
        ctx.KTStrategy.run_model()
    elif model_name == 'SAINT':

        ctx.set_model(SAINT)
        ctx.show_model()
        ctx.KTStrategy.set_data_name(data_name)
        print(ctx.KTStrategy.data_name)
        ctx.KTStrategy.run_model()
    elif model_name == 'SAKT':

        ctx.set_model(SAKT)
        ctx.show_model()
        ctx.KTStrategy.set_data_name(data_name)
        print(ctx.KTStrategy.data_name)
        ctx.KTStrategy.run_model()


    elif model_name == 'SAKT1e':

        ctx.set_model(SAKT_P1)
        ctx.show_model()
        ctx.KTStrategy.set_data_name(data_name)
        print(ctx.KTStrategy.data_name)
        ctx.KTStrategy.run_model()

    elif model_name == 'SAKT2e':

        ctx.set_model(SAKT_P2)
        ctx.show_model()
        ctx.KTStrategy.set_data_name(data_name)
        print(ctx.KTStrategy.data_name)
        ctx.KTStrategy.run_model()

    elif model_name == 'SAKT3e':

        ctx.set_model(SAKT_P3)
        ctx.show_model()
        ctx.KTStrategy.set_data_name(data_name)
        print(ctx.KTStrategy.data_name)
        ctx.KTStrategy.run_model()

    elif model_name == 'SAKTM':

        ctx.set_model(SAKT_P3I)
        ctx.show_model()
        ctx.KTStrategy.set_data_name(data_name)
        print(ctx.KTStrategy.data_name)
        ctx.KTStrategy.run_model()

    elif model_name == 'SAKTM':

        ctx.set_model(SAKTM)
        ctx.show_model()
        ctx.KTStrategy.set_data_name(data_name)
        print(ctx.KTStrategy.data_name)
        ctx.KTStrategy.run_model()
    return


# The first strategy: test 1 model with all dataset
def strategy2():
    rg = resultGenerator()
    print("test 1 model with all datasets")
    ctx = Context()
    ctx.set_model(KTStrategy)
    print("type the model you prefer,", MODEL_NAME)
    model_name = input()
    # model_name = 'DKT'
    # model_name = 'DKVMN'
    # model_name = 'DKTP'
    print("all dataset will be tested one by one", DATASET_NAME)
    if model_name == 'DKT':
        ctx.set_model(DKT)
        ctx.show_model()
    elif model_name == 'DKVMN':
        ctx.set_model(DKVMN)
        ctx.show_model()
    elif model_name == 'DKTP':
        ctx.set_model(DKTP)
        ctx.show_model()
    elif model_name == 'KQN':
        ctx.set_model(KQN)
        ctx.show_model()
    elif model_name == 'SAINT':
        ctx.set_model(SAINT)
        ctx.show_model()
    elif model_name == 'SAKT':
        ctx.set_model(SAKT)
        ctx.show_model()

    for data_name in DATASET_NAME:
        ctx.KTStrategy.set_data_name(data_name)
        print(ctx.KTStrategy.data_name)
        ctx.KTStrategy.run_model()
        rg.modeldata.append(model_name + " " + data_name)
        rg.auc.append(ctx.KTStrategy.re)
    rg.plot()


    return


# The first strategy: test all model with 1 dataset
def strategy3():
    rg = resultGenerator()
    print("test all models with 1 dataset")
    ctx = Context()
    ctx.set_model(KTStrategy)

    print("type the dataset you prefer,", DATASET_NAME)
    data_name = input()

    for m in MODEL_NAME:
        if m == 'DKT':
            ctx.set_model(DKT)
            ctx.show_model()
        elif m == 'DKVMN':
            ctx.set_model(DKVMN)
            ctx.show_model()

        elif m == 'DKTP':
            ctx.set_model(DKTP)
            ctx.show_model()

        elif m == 'KQN':
            ctx.set_model(KQN)
            ctx.show_model()
        elif m == 'SAINT':
            ctx.set_model(SAINT)
            ctx.show_model()
        elif m == 'SAKT':
            ctx.set_model(SAKT)
            ctx.show_model()

        ctx.KTStrategy.set_data_name(data_name)
        print(ctx.KTStrategy.data_name)
        ctx.KTStrategy.run_model()



        rg.modeldata.append(m + " " + data_name)
        rg.auc.append(ctx.KTStrategy.re)

        rg.plot()

    # run strategy 1 for each model
    # To do
    return


# The first strategy: test all model with all dataset
def strategy4():
    rg = resultGenerator()
    print("test all models with all datasets")
    ctx = Context()
    ctx.set_model(KTStrategy)

    for m in MODEL_NAME:
        if m == 'DKT':
            ctx.set_model(DKT)
            ctx.show_model()
        elif m == 'DKVMN':
            ctx.set_model(DKVMN)
            ctx.show_model()

        elif m == 'DKTP':
            ctx.set_model(DKTP)
            ctx.show_model()

        elif m == 'KQN':
            ctx.set_model(KQN)
            ctx.show_model()
        elif m == 'SAINT':
            ctx.set_model(SAINT)
            ctx.show_model()
        elif m == 'SAKT':
            ctx.set_model(SAKT)
            ctx.show_model()

        for data_name in DATASET_NAME:
            ctx.KTStrategy.set_data_name(data_name)
            print(ctx.KTStrategy.data_name)
            ctx.KTStrategy.run_model()
            rg.modeldata.append(m + " " + data_name)
            rg.auc.append(ctx.KTStrategy.re)


    rg.plot()
    # run strategy 2 for each model
    # To do
    return


# def test():
#     rg = resultGenerator()
#     rg.modeldata.append("DKVMN" + " " + "ASS09")
#     rg.auc.append(0.71)
#     rg.modeldata.append("DKVMN" + " " + "ASS12")
#     rg.auc.append(0.59)
#     rg.modeldata.append("DKVMN" + " " + "ASS15")
#     rg.auc.append(0.61)
#     rg.modeldata.append("DKVMN" + " " + "ASSCH")
#     rg.auc.append(0.60)
#     rg.modeldata.append("DKVMN" + " " + "STAT")
#     rg.auc.append(0.78)
#     rg.modeldata.append("DKVMN" + " " + "KDD05")
#     rg.auc.append(0.90)
#     rg.plot_line()

if __name__ == '__main__':
    # test()
    print("You can use this benchmark in four different ways:\n",
          "[1] The first is to test 1 model with 1 dataset,\n",
          "[2] The second is to test 1 model with all datasets,\n",
          "[3] The third is to test all models with 1 datasets,\n",
          "[4] The fourth is to test all models with all dataset.\n",
          "Input number 1, 2, 3, 4 to select your preferred approach to use this benchmark")
    # approach = input()
    approach = '1'
    if approach == '1':
        strategy1()

    elif approach == '2':
        strategy2()

    elif approach == '3':
        strategy3()

    elif approach == '4':
        strategy4()

    else:
        print("Select a valid approach from 1, 2, 3, 4")
        exit(1)
