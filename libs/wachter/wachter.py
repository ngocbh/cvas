import time

import torch.optim as optim
import torch

from torch.autograd import Variable
from sklearn.utils import check_random_state



def reconstruct_encoding_constraints(x, cat_pos):
    x_enc = x.clone()
    for pos in cat_pos:
        x_enc.data[pos] = torch.clamp(torch.round(x_enc[pos]), 0, 1)
    return x_enc


class Wachter(object):
    """ Class for generate counterfactual samples for framework: Wachter """
    DECISION_THRESHOLD = 0.5

    def __init__(self, model, cat_indices=list(), y_target=1, lambda_=0.1,
                 lr=0.01, dist_type=1, max_iter=1000, encoding_constraints=True):
        self.model = model
        self.lambda_ = lambda_
        self.lr = lr
        self.dist_type = dist_type
        self.max_iter = max_iter
        self.y_target = y_target
        self.cat_indices = cat_indices
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        self.feasible = True
        self.encoding_constraints = encoding_constraints

    def fit_instance(self, x0, threshold=None):
        threshold = threshold or Wachter.DECISION_THRESHOLD
        x0 = torch.from_numpy(x0.copy()).float().to(self.device)
        x = Variable(x0.clone(), requires_grad=True)
        x_enc = reconstruct_encoding_constraints(x, self.cat_indices)
        y_target = torch.tensor(self.y_target).float().to(self.device)
        lamb = torch.tensor(self.lambda_).float().to(self.device)
        f_x = self.model(x)

        loss_fn = torch.nn.BCELoss()
        optimizer = optim.Adam([x], self.lr, amsgrad=True)

        glob_it = 0
        while f_x <= threshold:
            it = 0
            while f_x <= threshold and it < self.max_iter:
                optimizer.zero_grad()

                if self.encoding_constraints:
                    x_enc = reconstruct_encoding_constraints(x, self.cat_indices)
                else:
                    x_enc = x.clone()

                f_x = self.model(x_enc).squeeze()

                cost = torch.dist(x_enc, x0, self.dist_type)
                f_loss = loss_fn(f_x, y_target)

                loss = f_loss + lamb * cost
                loss.backward()
                optimizer.step()
                it += 1

            lamb *= 0.5

            f_x = self.model(x_enc).squeeze()

            if glob_it >= 10:
                break
            glob_it += 1

        self.feasible = (f_x.data.item() > threshold)

        return x_enc.cpu().detach().numpy().squeeze()


def generate_recourse(x0, model, random_state, params=dict()):
    rng = check_random_state(random_state)

    train_data = params['train_data']
    ec = params['config']
    cat_indices = params['cat_indices']
    lambda_ = ec.wachter_params['lambda']
    threshold_shift = ec.threshold_shift

    start_time = time.time()

    arg = Wachter(model, cat_indices=cat_indices, y_target=1,
                  lambda_=lambda_, lr=0.01, dist_type=1, max_iter=1000)

    x_ar = arg.fit_instance(x0, threshold=0.5 + threshold_shift)

    runtime = time.time() - start_time
    report = dict(feasible=arg.feasible, runtime=runtime)

    return x_ar, report
