from tqdm import tqdm

import numpy as np

import recourse as rs


def add_actionability_constraints(acset, dataset_name):
    if dataset_name == 'synthesis':
        pass
    elif dataset_name == 'german':
        # set personal_status_and_sex feature as immutable
        acset['3'].actionable = False
        acset['4'].actionable = False
        acset['5'].actionable = False
        acset['6'].actionable = False
        # only increase age, only consider integer
        acset['2'].step_direction = 1 
        # acset['2'].step_size = 1
        # acset['2'].bound = (0, 100)
        # acset['2'].step_type = "absolute"
    elif dataset_name == 'sba':
        # Set urbanRural as immutable
        # only use Recession(7), New(3), RealEstate(6), Portion(5), Term(11), CreateJob(1)
        # acset['0'].actionable = False
        # acset['2'].actionable = False
        # acset['4'].actionable = False
        # acset['8'].actionable = False
        # acset['9'].actionable = False
        # acset['10'].actionable = False
        # acset['11'].actionable = False
        acset['12'].actionable = False
        acset['13'].actionable = False
        acset['14'].actionable = False
    elif dataset_name == 'student':
        # only increase age
        acset['5'].step_direction = 1
        # 
        acset['16'].actionable = False
        acset['17'].actionable = False
    elif dataset_name == 'twc':
        # only increase age
        acset['2'].step_direction = 1
        # 
        acset['20'].actionable = False
        acset['21'].actionable = False
        acset['22'].actionable = False
        acset['23'].actionable = False
        acset['24'].actionable = False
        acset['25'].actionable = False
        acset['26'].actionable = False
        acset['27'].actionable = False
        acset['28'].actionable = False
        acset['29'].actionable = False
        acset['30'].actionable = False
        acset['31'].actionable = False
        acset['32'].actionable = False
    else:
        raise ValueError("Do not support dataset: {}".format(dataset_name))



class LinearAR(object):
    """ Class for generate counterfactual samples for framework: AR """

    def __init__(self, data, coef, intercept, dataset_name=None):
        """ Parameters
        Args:
            data: data to get upper bound, lower bound, action set
            coef: coefficients of classifier
            intercept: intercept of classifier
        """
        self.n_variables = data.shape[1]

        # Action set
        name_l = [str(i) for i in range(self.n_variables)]
        self.action_set = rs.ActionSet(data, names=name_l)
        self.coef = coef
        self.intercept = intercept
        if dataset_name is not None:
            add_actionability_constraints(self.action_set, dataset_name)

        self.action_set.set_alignment(coefficients=coef, intercept=intercept)

        self.feasible = True

    def fit_instance(self, x):
        """ Fit linear recourse action with an instance
        Args:
            x: a single input
        Returns:
            counterfactual_sample: counterfactual of input x
        """
        self.feasible = None
        try:
            rb = rs.RecourseBuilder(
                coefficients=self.coef,
                intercept=self.intercept,
                action_set=self.action_set,
                x=x
            )
            output = rb.fit()
            if output['feasible']:
                self.feasible = True
                counterfactual_sample = np.add(x, output['actions'])
            else:
                self.feasible = False
                counterfactual_sample = x
        except:
            self.feasible = False
            counterfactual_sample = x

        return counterfactual_sample

    def fit_data(self, data):
        """ Fit linear recourse action with all instances
        Args:
            data: all the input instances
        Returns:
            counterfactual_samples: counterfactual of instances in dataset
        """
        l = len(data)
        counterfactual_samples = np.zeros((l, self.n_variables))

        for i in tqdm(range(l)):
            counterfactual_samples[i] = self.fit_instance(data[i])

        return counterfactual_samples
