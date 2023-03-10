from hmm import *

# Part IV
# ---------


class UpdatedSuspectState(IntEnum):
    # Add custom states here
    # pass
    Planning = 1,
    Scouting = 2,
    Burglary = 3,
    Hiding = 4
    Migrating = 5,
    Misc = 6

class Updated_HMM:
    # Complete this implementation
    # for part IV of the assignment
    # pass
    def __init__(self) -> None:
        self.trans_mat = np.array([
        [1/3,1/3,0,0,0,1/3],
        [0,1/3,1/3,0,0,1/3],
        [0,0,0,1,0,0],
        [1,0,0,0,0,0],
        [1/3,1/3,0,0,0,1/3],
        [1/4,1/4,0,1/4,0,1/4]
        ])
        
        self.pi = np.array([0,0,0.5,0.5,0])
        
        self.emission_mat = np.array([
        [0,0,1/3,0,0,0,1/3,0,0,0,1/3,0],
        [0.2,0,0,0,0.2,0,0,0,0.6,0,0,0],
        [0,0,0,1/3,0,0,0,1/3,0,0,0,1/3],
        [0,0,0,1/3,0,0,0,1/3,0,0,0,1/3],
        [0,0.2,0,0,0,0.6,0,0,0,0.2,0,0],
        [0,0.2,0,0.2,0,0.4,0,0,0,0.2,0,0]
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

# Since python does not support function 
# overloading (unlike C/C++), the names of 
# the following functions are made to be different

# Reads the dataset of array of sequence of observation
# and initializes a HMM model from it.
# returns Initialized HMM.
# Here, the parameter `dataset` is
# a list of list of `Observation` class.
# Each (inner) list represents the sequence of observation
# from start to end as mentioned in the question


def LearnUpdatedModel(dataset: list) -> Updated_HMM:
    dataset_nums  = ConvertDatasetToIdx(dataset)
    # dataset_nums = np.random.random_integers(low=0,high=12,size=(24,12))
    # print("length of dataset:",len(dataset))
    hmm_obj_updated = Updated_HMM()
    # print("shape:",dataset_nums.shape)
    # for i in dataset:
        # print(len(i))
    #     for j in i:
    #         print(j)
    #         x = np.expand_dims(np.array([j.daytime,j.action]),axis=0)
    #         print(x)
    # for i in dataset_nums:
            # x = np.expand_dims(np.array(i),axis=0)
            # hmm_obj_updated.model.fit(x)
    hmm_obj_updated.model.fit(dataset_nums)
    hmm_obj_updated.trans_mat = hmm_obj_updated.model.transmat_
    hmm_obj_updated.pi = hmm_obj_updated.model.get_stationary_distribution()
    # print(*hmm_obj_updated.trans_mat,sep=",\n")
    # print(hmm_obj_updated.pi)
    return hmm_obj_updated


# Given an initialized HMM model,
# and some set of observations, this function evaluates
# the liklihood that this set of observation was indeed
# generated from the given model.
# Here, the obs_list is a list containing
# instances of the `Observation` class.
# The output returned has to be floatint point between
# 0 and 1


def LiklihoodUpdated(model: Updated_HMM, obs_list: list) -> float:
    # pass
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

# Given an initialized model, and a sequence of observation,
# returns  a list of the same size, which contains
# the most likely # states the model
# was in to produce the given observations.
# returns An array/sequence of states that the model was in to
# produce the corresponding sequence of observations

def GetUpdatedHiddenStates(model: Updated_HMM, obs_list: list) -> list:
    # pass
    x_input = ConvertDatasetToIdx([obs_list])
    # print(x_input)
    liklihood_involve = model.model.decode(x_input)
    State_List = [UpdatedSuspectState.Planning, UpdatedSuspectState.Scouting,
              UpdatedSuspectState.Burglary, UpdatedSuspectState.Hiding,UpdatedSuspectState.Migrating, UpdatedSuspectState.Misc]
    # print(liklihood_involve[1])
    updated_suspectstate_list = []
    for i in liklihood_involve[1]:
        updated_suspectstate_list.append(State_List[i])
    return updated_suspectstate_list

if __name__ == "__main__":
    database = ReadDataset()
    old_model = LearnModel(database)
    new_model = LearnUpdatedModel(database)


    # obs_list = [ ] # Add your list of observations
    obs_list = [
        Observation(Daytime.Day, Action.Home),
        Observation(Daytime.Evening, Action.Eating),
        Observation(Daytime.Night, Action.Home),
        Observation(Daytime.Day, Action.Home),
        Observation(Daytime.Day, Action.Untracked),
        Observation(Daytime.Evening, Action.Eating),
        Observation(Daytime.Evening, Action.Untracked),
        Observation(Daytime.Night, Action.Roaming),
        Observation(Daytime.Night, Action.Untracked),
    ]
    p = Liklihood(old_model, obs_list)
    q = Liklihood(new_model, obs_list)

    old_states = GetHiddenStates(old_model, obs_list)
    new_states = GetUpdatedHiddenStates(new_model, obs_list)

    # Add code to showcase and compare the obtained
    # results between the two models
    print("old:",p,"new:",q)
    print("old states:",old_states)
    print("new states:",new_states)
