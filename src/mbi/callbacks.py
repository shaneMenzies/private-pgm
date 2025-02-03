import attr
from collections import deque
import jax
from mbi import LinearMeasurement, Dataset, CliqueVector
from mbi import marginal_loss
import pandas as pd
import chex


def _pad(string: str, length: int):
    if len(string) > length:
        return string[:length]
    left_pad = (length - len(string)) // 2
    right_pad = length - len(string) - left_pad
    return " " * left_pad + string + " " * right_pad


@attr.dataclass
class Callback:
    loss_fns: dict[str, marginal_loss.MarginalLossFn]
    frequency: int = 50
    history_len: int = 25
    margin_percent: float = 1e-9
    history: deque = deque()

    # Internal state
    _step: int = 0
    _logs: list = attr.field(factory=list)

    def __call__(self, marginals: CliqueVector) -> bool:
        if self._step == 0:
            header = "|".join([_pad(x, 12) for x in ["step", *self.loss_fns.keys()]])
            print(header)
            print("=" * len(header))
        
        loss_vals: dict[str, chex.Numeric] = {}
        for key in self.loss_fns:
            loss_vals[key] = self.loss_fns[key](marginals)

        if self._step % self.frequency == 0:
            row = [loss_vals[key] for key in loss_vals]
            self._logs.append([self._step] + row)
            padded_step = str(self._step) + " " * (9 - len(str(self._step)))
            print(padded_step, *[("%.6f" % v)[:6] for v in row], sep="   |   ")
        self._step += 1

        self.history.append(loss_vals)
        if len(self.history) > self.history_len:
            self.history.popleft()
            for key in loss_vals:
                margin = self.margin_percent * loss_vals[key]
                diff = self.history[0][key] - loss_vals[key]
                if diff > margin:
                    return False
            print("Stopping early due to lack of improvement (after", self._step, "iterations)")
            row = [loss_vals[key] for key in loss_vals]
            self._logs.append([self._step] + row)
            padded_step = str(self._step) + " " * (9 - len(str(self._step)))
            print(padded_step, *[("%.6f" % v)[:6] for v in row], sep="   |   ")
            return True
        else:
            return False

        

    @property
    def summary(self):
        return pd.DataFrame(
            columns=["step"] + list(self.loss_fns.keys()), data=self._logs
        ).astype(float)


def default(
    measurements: list[LinearMeasurement],
    data: Dataset | None = None,
    frequency: int = 50,
) -> Callback:
    loss_fns = {}
    # Measures distance between input marginals and noisy marginals.
    loss_fns["L2 Loss"] = marginal_loss.from_linear_measurements(
        measurements, norm="l2", normalize=True
    )
    loss_fns["L1 Loss"] = marginal_loss.from_linear_measurements(
        measurements,
        norm="l1",
        normalize=True,
    )

    if data is not None:
        # Measures distance between input marginals and true marginals.
        ground_truth = [
            LinearMeasurement(
                M.query(data.project(M.clique).datavector()),
                clique=M.clique,
                stddev=1,
                query=M.query,
            )
            for M in measurements
        ]
        loss_fns["L2 Error"] = marginal_loss.from_linear_measurements(
            ground_truth, norm="l2", normalize=True
        )
        loss_fns["L1 Error"] = marginal_loss.from_linear_measurements(
            ground_truth, norm="l1", normalize=True
        )

    loss_fns = {key: jax.jit(loss_fns[key].__call__) for key in loss_fns}
    loss_fns["Primal Feas"] = jax.jit(marginal_loss.primal_feasibility)

    return Callback(loss_fns, frequency)
