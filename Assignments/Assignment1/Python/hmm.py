from enum import IntEnum 
import numpy as np
import os

class SuspectState(IntEnum):
    Planning = 1,
    Scouting = 2,
    Burglary = 3,
    Migrating = 4,
    Misc = 5

class Daytime(IntEnum):
    Day = 6
    Evening = 7
    Night = 8

class Action(IntEnum):
    Roaming = 9,
    Eating = 10,
    Home = 11,
    Untracked = 12,

class Observation:
    def __init__(self, d:Daytime, a:Action) -> None:
        self.daytime = d
        self.action = a


# This function reads the string file 
# that contains the sequence of observations
# that is required to learn the HMM model.
def ReadDataset() -> list:
    # Converts integer to Daytime enum
    def getDay(p: int) -> Daytime:
        if p == 6:
            return Daytime.Day
        elif p == 7:
            return Daytime.Evening
        elif p == 8:
            return Daytime.Night
        else:
            assert False, 'Unexpected Daytime!'

    # Converts integer to Action enum
    def getAct(p: int) -> Action:
        if p == 9:
            return Action.Roaming
        elif p == 10:
            return Action.Eating
        elif p == 11:
            return Action.Home
        elif p == 12:
            return Action.Untracked
        else:
            assert False, 'Unexpected Action!'
    filepath = 'database.txt' 
    with open(filepath, 'r') as file:
        seq_count = int(file.readline())
        seq_list = []
        for _ in range(seq_count):
            w = file.readline().split(' ')
            len = int(w[0])
            seq_i = []
            for k in range(0, len):
                idx = (k*2) + 1
                day = int(w[idx])
                act = int(w[idx + 1])
                o = Observation(getDay(day), getAct(act))
                seq_i.append(o)
            seq_list.append(seq_i)
    return seq_list


#  --------------Do not change anything above this line---------------
class HMM:
    # Complete the HMM implementation.
    # The three function below must
    # be implemented.
    def __init__(self) -> None:
        self.trans_mat = np.array([
        [1/3,1/3,0,0,1/3],
        [0,1/3,1/3,0,1/3],
        [0,0,0,1,0],
        [1,0,0,0,0],
        [1/3,1/3,0,0,1/3]
        ])
        
        self.pi = np.array([0,0,0.5,0.5,0])
        
        self.emission_mat = np.array([
        [0,0,1/3,0,0,0,1/3,0,0,0,1/3,0],
        [0.2,0,0,0,0.2,0,0,0,0.6,0,0,0],
        [0,0,0,1/3,0,0,0,1/3,0,0,0,1/3],
        [0,0,0,1/3,0,0,0,1/3,0,0,0,1/3],
        [0,0.2,0,0,0,0.6,0,0,0,0.2,0,0]
        ])
        
        try:
            from hmmlearn.hmm import CategoricalHMM
        except:    
            os.system("pip install -q hmmlearn")
            # pip.main(['install', 'hmmlearn'])
            print("hmmlearn not installed!!!")
        finally:
            from hmmlearn.hmm import CategoricalHMM
        
        self.model = CategoricalHMM(
            random_state=42,
            n_components=len(self.trans_mat[0]),
            emissionprob_prior=self.emission_mat,
            transmat_prior=self.trans_mat,
            startprob_prior=self.pi,
            algorithm="viterbi",
            params="et",
            )

        # self.model.n_features = 12
        # self.model.startprob_ = self.pi
        # self.model.transmat_ = self.trans_mat
        # self.model.emissionprob_ = self.emission_mat
        # set.model = MultinomialHMM(n_components=len(self.trans_mat[0]),startprob_prior=self.pi,)
    
    def A(self, a: SuspectState, b: SuspectState) -> float:
        # Compute the probablity of going from one
        # state a to the other state b
        # Hint: The necessary code does not need
        # to be within this function only
        # pass
        return self.trans_mat[a-1][b-1]

    def B(self, a: SuspectState, b: Observation) -> float:
        # Compute the probablity of obtaining
        # observation b from state a
        # Hint: The necessary code does not need
        # to be within this function only
        # pass
        # print(b)
        return self.emission_mat[a-1][ConvertTupleToInt(b)]

    def Pi(self, a: SuspectState) -> float:
        # Compute the initial probablity of
        # being at state a
        # Hint: The necessary code does not need
        # to be within this function only
        # pass
        return self.pi[a-1]
    
# user defined function
def ConvertTupleToInt(data_tuple):
    i = data_tuple
    data = []
    if i.daytime == Daytime.Day:
        if i.action == Action.Roaming:
            data.append(0)
        elif i.action == Action.Eating:
            data.append(1)
        elif i.action == Action.Home:
            data.append(2)
        elif i.action == Action.Untracked:
            data.append(3)
            
    elif i.daytime == Daytime.Evening:
        if i.action == Action.Roaming:
            data.append(4)
        elif i.action == Action.Eating:
            data.append(5)
        elif i.action == Action.Home:
            data.append(6)
        elif i.action == Action.Untracked:
            data.append(7)

    elif i.daytime == Daytime.Night:
        if i.action == Action.Roaming:
            data.append(8)
        elif i.action == Action.Eating:
            data.append(9)
        elif i.action == Action.Home:
            data.append(10)
        elif i.action == Action.Untracked:
            data.append(11)
    # print(data)
    return data[0]
    
    
def ConvertDatasetToIdx(seq_list: list) -> list:
    # print(seq_list)
    new_data = []
    max_len = 0
    for i in seq_list:
        data = []
        count = 0
        
        while count < len(i):
            if i[count].daytime == Daytime.Day:
                if i[count].action == Action.Roaming:
                    data.append(0)
                elif i[count].action == Action.Eating:
                    data.append(1)
                elif i[count].action == Action.Home:
                    data.append(2)
                elif i[count].action == Action.Untracked:
                    data.append(3)
            
            elif i[count].daytime == Daytime.Evening:
                if i[count].action == Action.Roaming:
                    data.append(4)
                elif i[count].action == Action.Eating:
                    data.append(5)
                elif i[count].action == Action.Home:
                    data.append(6)
                elif i[count].action == Action.Untracked:
                    data.append(7)

            elif i[count].daytime == Daytime.Night:
                if i[count].action == Action.Roaming:
                    data.append(8)
                elif i[count].action == Action.Eating:
                    data.append(9)
                elif i[count].action == Action.Home:
                    data.append(10)
                elif i[count].action == Action.Untracked:
                    data.append(11)
            count = count + 1
        max_len = max(max_len,len(i))
        new_data.append(data)

    # print(new_data)
    # print(max_len)
    for i in new_data:
        if len(i) < max_len:
            np.random.seed(42)
            i.extend(np.random.random_integers(low=0,high=11,size=(max_len-len(i))))
    # print(*new_data,sep="\n")
    return np.array(new_data)
    return new_data

# Part I
# ---------

# Reads the dataset of array of sequence of observation
# and initializes a HMM model from it.
# returns Initialized HMM.
# Here, the parameter `dataset` is
# a list of list of `Observation` class.
# Each (inner) list represents the sequence of observation
# from start to end as mentioned in the question


def LearnModel(dataset: list) -> HMM:
    # pass
    # for data in dataset:
        # for item in data:
            # print(item.daytime,item.action,end=" ")
            # pass
        # print("\n")
    dataset_nums  = ConvertDatasetToIdx(dataset)
    # dataset_nums = np.random.random_integers(low=0,high=12,size=(24,12))
    # print("length of dataset:",len(dataset))
    hmm_obj = HMM()
    # print("shape:",dataset_nums.shape)
    # for i in dataset:
        # print(len(i))
    #     for j in i:
    #         print(j)
    #         x = np.expand_dims(np.array([j.daytime,j.action]),axis=0)
    #         print(x)
    # for i in dataset_nums:
            # x = np.expand_dims(np.array(i),axis=0)
            # hmm_obj.model.fit(x)
    hmm_obj.model.fit(dataset_nums)
    hmm_obj.trans_mat = hmm_obj.model.transmat_
    hmm_obj.pi = hmm_obj.model.get_stationary_distribution()
    # print(*hmm_obj.trans_mat,sep=",\n")
    # print(hmm_obj.pi)
    return hmm_obj

# Part II
# ---------

# Given an initialized HMM model,
# and some set of observations, this function evaluates
# the liklihood that this set of observation was indeed
# generated from the given model.
# Here, the obs_list is a list containing
# instances of the `Observation` class.
# The output returned has to be floatint point between
# 0 and 1

# def forward(O, a, b, initial_distribution):
#     print("In Forward Function")
#     alpha = np.zeros((O.shape[0], a.shape[0]))
#     print(alpha.shape)
#     # alpha[0, :] = initial_distribution * b[:, O[0]]
#     print(O[0].daytime,O[0].action)
#     print(b)
#     print(initial_distribution)
#     print(alpha)
#     for t in range(1, O.shape[0]):
#         for j in range(a.shape[0]):
#             alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, O[t]]
#     print("End Forward function")
#     return alpha

def Liklihood(model: HMM, obs_list: list) -> float:
    # pass
    # print(type(obs_list))
    # for obs in obs_list:
    #     print(obs.daytime,obs.action,end=" ")
    # print("------------------")
    # probs = forward(np.array(obs_list),model.trans_mat,model.emission_mat,model.pi)
    # print(probs)
    x_input = ConvertDatasetToIdx([obs_list])
    liklihood_involve = model.model.score(x_input)
    # print(liklihood_involve)
    # print(np.e ** liklihood_involve)
    # print(np.exp(liklihood_involve))
    # print(np.average(liklihood_involve,axis=1))
    # return np.average(liklihood_involve)
    # return np.exp(liklihood_involve)
    return np.e ** liklihood_involve

# Part III
# ---------

# Given an initialized model, and a sequence of observation,
# returns  a list of the same size, which contains
# the most likely # states the model
# was in to produce the given observations.
# returns An array/sequence of states that the model was in to
# produce the corresponding sequence of observations

def GetHiddenStates(model: HMM, obs_list: list) -> list:
    # pass
    # for obs in obs_list:
    #     print(obs.daytime,obs.action)
    # print("------------------")
    x_input = ConvertDatasetToIdx([obs_list])
    # print(x_input)
    liklihood_involve = model.model.decode(x_input)
    State_List = [SuspectState.Planning, SuspectState.Scouting,
              SuspectState.Burglary, SuspectState.Migrating, SuspectState.Misc]
    # print(liklihood_involve[1])
    suspectstate_list = []
    for i in liklihood_involve[1]:
        suspectstate_list.append(State_List[i])
    return suspectstate_list
