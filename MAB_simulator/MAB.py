from typing import Dict,Callable,List
from scipy.stats import bernoulli

global_last_arm_id = 0  # Last assigned arm_id

class Arm:
    def __init__(self,id):
        self.__arm_id = id
        self.reward_func = lambda : 0
        self._is_initialized = False

    def initialize(self,reward_function : Callable[[Dict],float]):
        ...

    def pull(self,args : Dict):
        ...

    def get_arm_id(self):
        return self.__arm_id

class BernoulliArm(Arm):
    def __init__(self, id:int ,success_prob:float):
        super().__init__(id=id)
        if not 0 <= success_prob <= 1:
            raise ValueError("Arm {id} success probability should be in [0,1] , but it is {prob} ".format(id=id,prob=success_prob))
        self.inner_random_var = bernoulli(success_prob)

    def initialize(self,reward_function : Callable[[Dict],float]):
        if self._is_initialized :
            raise Exception("Attemp to initialize arm {} twice".format(self.get_arm_id()))
        self.reward_func = reward_function
        self._is_initialized = True

    def pull(self,args : Dict):
        if not self._is_initialized :
            raise Exception("Attemp to play uninitialized arm {}".format(self.get_arm_id()))
        args.update({"inner_random_var" : self.inner_random_var})
        return self.reward_func(args)

def one_zero_bernoulli_reward(args):

    #TODO warning if there more than 1 args
    return args["inner_random_var"].rvs()


arm1 = BernoulliArm(1,0.5)
arm1.initialize(one_zero_bernoulli_reward)
for i in range(10):
    print(arm1.pull({}))


def random_bernoulli_arm_factory(args : Dict):
    arms_number = args["arms_number"]


class Bandit:
    def __init__(self,id):
        self.bandit_id = id
        self.__arms_list = []  #type: List[Arm]
        self.cumulative_regret = 0
        self.current_order = [] #type: List[Arm]      # The order in which the armes are pulled if function play is called
        self.round_number = 0
        self._is_initialized = False


        self.min_arm_id = None       # The minimal id of all the arms that this bandit has - used for simple reordering of the arms
        self.last_regret = None

    def initialize(self,arm_init_function : Callable[[Dict],List[Arm]] , args : Dict):
        if self._is_initialized :
            raise Exception("Attemp to initialize bandit {} twice".format(self.bandit_id))
        init_bandit_args = {}
        # Create arms from given factory
        self.__arms_list = arm_init_function(args)

        #TODO : check if the order is correct - integers in ascending order like in range(n,m)

        # Create initial order
        self.min_arm_id = self.__arms_list[0].get_arm_id()       # The first arm in the list must have the lowest ID !
        self.current_order = list(self.__arms_list)


    def set_order(self,arms_order : List[int]):

        #TODO : Check if the indexes in order are correct with respect to arms we have

        # Convert the ordered list of arm ids to ordered list of corresponding indexes in self.__arms_list
        arm_indexes = [(arm_id - self.min_arm_id) for arm_id in arms_order]

        self.current_order = [self.__arms_list[i] for i in arm_indexes]

    def __update_relevant_info(self,data : Dict):

        #TODO think how I compute \" best possible round \" rewards ( to compute the regret )
        self.last_regret = data["round_reward"]

        self.cumulative_regret += self.last_regret


    def play(self,args: Dict=None):
        round_reward = 0

        # Pull the arms until failure
        for arm in self.current_order:
            arm_reward = arm.pull(args)
            if arm_reward == 0:
                break
            round_reward += arm_reward

        # Update relevant inner statistics/data of the bandit ( Regret etc.)
        self.__update_relevant_info({"round_reward" : round_reward})

        return round_reward
