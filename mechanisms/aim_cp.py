"""Implementation of AIM: An Adaptive and Iterative Mechanism for DP Synthetic Data.

Note that with the default settings, AIM can take many hours to run.  You can configure
the runtime /utility tradeoff via the max_model_size flag.  We recommend setting it to 1.0
for debugging, but keeping the default value of 80 for any official comparisons to this mechanism.

Note that we assume in this file that the data has been appropriately preprocessed so that there are no large-cardinality categorical attributes.  If there are, we recommend using something like "compress_domain" from mst.py.  Since our paper evaluated already-preprocessed datastes, we did not implement that here for simplicity.
"""

import jax.numpy as np
import itertools
from mbi import (
    callbacks,
    Dataset,
    Domain,
    estimation,
    marginal_oracles,
    junction_tree,
    LinearMeasurement,
    LinearMeasurement,
)
from mechanism import Mechanism
from collections import defaultdict
from scipy.optimize import bisect
import pandas as pd
from mbi import Factor
import argparse
import dill
import jax

def save_object(obj, filename):
    with open(filename, "wb") as output:
        dill.dump(obj, output)

def load_object(filename):
    with open(filename, "rb") as input:
        return dill.load(input)


def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(1, len(s) + 1)
    )


def downward_closure(Ws):
    ans = set()
    for proj in Ws:
        ans.update(powerset(proj))
    return list(sorted(ans, key=len))

def compile_workload(workload):
    weights = {cl: wt for (cl, wt) in workload}
    workload_cliques = weights.keys()

    def score(cl):
        return sum(
            weights[workload_cl] * len(set(cl) & set(workload_cl))
            for workload_cl in workload_cliques
        )

    return {cl: score(cl) for cl in downward_closure(workload_cliques)}


def filter_candidates(candidates, model, size_limit):
    ans = {}
    free_cliques = downward_closure(model.cliques)
    for cl in candidates:
        # cond1 = (
        #    hypothetical_model_size(model.domain, model.cliques + [cl]) <= size_limit
        #)
        cond2 = cl in free_cliques
        if cond2:
            ans[cl] = candidates[cl]
    return ans


class AIM(Mechanism):
    def __init__(
        self,
        epsilon,
        delta,
        prng=None,
        rounds=None,
        max_model_size=80,
        max_iters=1000,
        structural_zeros={},
    ):
        super(AIM, self).__init__(epsilon, delta, prng)
        self.rounds = rounds
        self.max_iters = max_iters
        self.max_model_size = max_model_size
        self.structural_zeros = structural_zeros

    def worst_approximated(self, candidates, answers, model, eps, sigma):
        errors = {}
        sensitivity = {}
        for cl in candidates:
            wgt = candidates[cl]
            x = answers[cl]
            bias = np.sqrt(2 / np.pi) * sigma * model.domain.size(cl)
            xest = model.project(cl).datavector()
            errors[cl] = wgt * (np.linalg.norm(x - xest, 1) - bias)
            sensitivity[cl] = abs(wgt)

        max_sensitivity = max(
            sensitivity.values()
        )  # if all weights are 0, could be a problem
        return self.exponential_mechanism(errors, eps, max_sensitivity)

    def run(self, data, workload, chkpnt_loc, num_synth_rows=None, initial_cliques=None):
        self.data = data
        self.num_synth_rows = num_synth_rows
        rounds = self.rounds or 16 * len(self.data.domain)
        self.candidates = compile_workload(workload)
        print("Compiled workload")
        self.answers = {cl: self.data.project(cl).datavector() for cl in self.candidates}
        print("Projected answers from candidates(?)")

        if not initial_cliques:
            initial_cliques = [
                cl for cl in self.candidates if len(cl) == 1
            ]  # use one-way marginals
        oneway = [cl for cl in self.candidates if len(cl) == 1]
        print("Determined initial cliques")

        self.sigma = np.sqrt(rounds / (2 * 0.9 * self.rho))
        self.epsilon = np.sqrt(8 * 0.1 * self.rho / rounds)

        self.measurements = []
        print("Initial Sigma", self.sigma)
        self.rho_used = len(oneway) * 0.5 / self.sigma**2
        for cl in initial_cliques:
            x = self.data.project(cl).datavector()
            y = x + self.gaussian_noise(self.sigma, x.size)
            self.measurements.append(LinearMeasurement(y, cl, stddev=self.sigma))
        print("Made initial measurements")

        # NOTE: Haven't incorproated structural zeros back yet after refactoring
        self.model = estimation.mirror_descent(
                self.data.domain, self.measurements, iters=self.max_iters, 
                marginal_oracle=marginal_oracles.message_passing_fast,
                callback_fn=callbacks.default(self.measurements, self.data)
        )
        print("Setup model")

        self.t = 0
        return self.run_train(chkpnt_loc)

    def run_train(self, chkpnt_loc):
        terminate = False
        while not terminate:
            self.t += 1
            if self.rho - self.rho_used < 2 * (0.5 / self.sigma**2 + 1.0 / 8 * self.epsilon**2):
                # Just use up whatever remaining budget there is for one last round
                remaining = self.rho - self.rho_used
                self.sigma = np.sqrt(1 / (2 * 0.9 * remaining))
                self.epsilon = np.sqrt(8 * 0.1 * remaining)
                terminate = True

            self.rho_used += 1.0 / 8 * self.epsilon**2 + 0.5 / self.sigma**2
            print('Budget Used', self.rho_used, '/', self.rho)
            self.size_limit = self.max_model_size * self.rho_used / self.rho

            small_candidates = filter_candidates(self.candidates, self.model, self.size_limit)
            cl = self.worst_approximated(
                small_candidates, self.answers, self.model, self.epsilon, self.sigma
            )
            print('Measuring Clique', cl)
            n = self.data.domain.size(cl)
            x = self.data.project(cl).datavector()
            y = x + self.gaussian_noise(self.sigma, n)
            self.measurements.append(LinearMeasurement(y, cl, stddev=self.sigma))
            z = self.model.project(cl).datavector()

            # Warm start potentials from prior round
            # TODO: check if it helps to call maximal_subsets here
            pcliques = list(set(M.clique for M in self.measurements))
            potentials = self.model.potentials.expand(pcliques)
            self.model = estimation.mirror_descent(
                    self.data.domain, self.measurements, iters=self.max_iters, 
                    marginal_oracle=marginal_oracles.message_passing_fast,
                    potentials=potentials, 
                    callback_fn=callbacks.default(self.measurements, self.data)
            )
            w = self.model.project(cl).datavector()
            # print('Selected',cl,'Size',n,'Budget Used',self.rho_used/self.rho)
            if np.linalg.norm(w - z, 1) <= self.sigma * np.sqrt(2 / np.pi) * n:
                print("(!!!!!!!!!!!!!!!!!!!!!!) Reducing self.sigma", self.sigma / 2)
                self.sigma /= 2
                self.epsilon *= 2

            save_object(self, chkpnt_loc)

        print("Generating Data...")
        self.model = estimation.mirror_descent(
            self.data.domain, self.measurements, iters=self.max_iters, potentials=potentials
        )
        synth = self.model.synthetic_data(rows=self.num_synth_rows)
        return self.model, synth


def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params["dataset"] = "../data/adult.csv"
    params["domain"] = "../data/adult-domain.json"
    params["epsilon"] = 1.0
    params["delta"] = 1e-9
    params["noise"] = "laplace"
    params["max_model_size"] = 80
    params["max_iters"] = 1000
    params["degree"] = 2
    params["num_marginals"] = None
    params["max_cells"] = 10000

    return params

def continue_from_checkpoint(checkpoint):
    model = load_object(checkpoint)

    return model.run_train(checkpoint)


if __name__ == "__main__":

    description = ""
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument("--dataset", help="dataset to use")
    parser.add_argument("--domain", help="domain to use")
    parser.add_argument("--epsilon", type=float, help="privacy parameter")
    parser.add_argument("--delta", type=float, help="privacy parameter")
    parser.add_argument(
        "--max_model_size", type=float, help="maximum size (in megabytes) of model"
    )
    parser.add_argument("--max_iters", type=int, help="maximum number of iterations")
    parser.add_argument("--degree", type=int, help="degree of marginals in workload")
    parser.add_argument(
        "--num_marginals", type=int, help="number of marginals in workload"
    )
    parser.add_argument(
        "--max_cells",
        type=int,
        help="maximum number of cells for marginals in workload",
    )
    parser.add_argument("--save", type=str, help="path to save synthetic data")

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    data = Dataset.load(args.dataset, args.domain)

    workload = list(itertools.combinations(data.domain, args.degree))
    workload = [cl for cl in workload if data.domain.size(cl) <= args.max_cells]
    if args.num_marginals is not None:
        prng = np.random
        workload = [
            workload[i]
            for i in prng.choice(len(workload), args.num_marginals, replace=False)
        ]

    workload = [(cl, 1.0) for cl in workload]
    mech = AIM(
        args.epsilon,
        args.delta,
        max_model_size=args.max_model_size,
        max_iters=args.max_iters,
    )
    model, synth = mech.run(data, workload)

    if args.save is not None:
        synth.df.to_csv(args.save, index=False)

    errors = []
    for proj, wgt in workload:
        X = data.project(proj).datavector()
        Y = synth.project(proj).datavector()
        e = 0.5 * wgt * np.linalg.norm(X / X.sum() - Y / Y.sum(), 1)
        errors.append(e)
    print("Average Error: ", np.mean(np.asarray(errors)))
