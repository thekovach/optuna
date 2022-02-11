from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Union

import optuna
from optuna._experimental import experimental
from optuna._imports import try_import


with try_import() as _imports:
    import comet_ml


@experimental("3.0.0")
class CometCallback(object):
    """Callback to track Optuna trials with Comet.ml.

    This callback enables tracking of Optuna study in
    Comet.ml. Every trial in a study is tracked as a single experiment,
    where all suggested hyperparameters and optimized metrics
    are logged.

    .. note::
        User needs to be have Comet.ml configured before
        using this callback in online mode. For more information, please
        refer to `comet_ml setup <https://www.comet.ml/docs/python-sdk/advanced/#python-configuration>`_.


    Example:

        Add Comet.ml callback to Optuna optimization.

        .. code::

            import optuna
            from optuna.integration.comet import CometCallback


            def objective(trial):
                x = trial.suggest_float("x", -10, 10)
                return (x - 2) ** 2


            comet_kwargs = {"project": "my-project"}
            cometc = CometCallback(comet_kwargs=comet_kwargs)


            study = optuna.create_study(study_name="my_study")
            study.optimize(objective, n_trials=10, callbacks=[cometc])

    Args:
        metric_name:
            Name assigned to optimized metric. In case of multi-objective optimization,
            list of names can be passed. Those names will be assigned
            to metrics in the order returned by objective function.
            If single name is provided, or this argument is left to default value,
            it will be broadcasted to each objective with a number suffix in order
            returned by objective function e.g. two objectives and default metric name
            will be logged as ``value_0`` and ``value_1``.
        comet_kwargs:
            Set of arguments passed when initializing Comet Experiement.
            Please refer to `Comet.ml API documentation
            <https://www.comet.ml/docs/python-sdk/experiment-overview/>`_ for more details.

    Raises:
        :exc:`ValueError`:
            If there are missing or extra metric names in multi-objective optimization.
        :exc:`TypeError`:
            When metric names are not passed as sequence.
    """

    def __init__(
        self,
        metric_name: Union[str, Sequence[str]] = "value",
        comet_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:

        _imports.check()

        if not isinstance(metric_name, Sequence):
            raise TypeError(
                "Expected metric_name to be string or sequence of strings, got {}.".format(
                    type(metric_name)
                )
            )

        self._metric_name = metric_name
        self._comet_kwargs = comet_kwargs or {}

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:

        if isinstance(self._metric_name, str):
            if len(trial.values) > 1:
                # Broadcast default name for multi-objective optimization.
                names = ["{}_{}".format(self._metric_name, i) for i in range(len(trial.values))]

            else:
                names = [self._metric_name]

        else:
            if len(self._metric_name) != len(trial.values):
                raise ValueError(
                    "Running multi-objective optimization "
                    "with {} objective values, but {} names specified. "
                    "Match objective values and names, or use default broadcasting.".format(
                        len(trial.values), len(self._metric_name)
                    )
                )

            else:
                names = [*self._metric_name]

        metrics = {name: value for name, value in zip(names, trial.values)}
        attributes = {"direction": [d.name for d in study.directions]}

        exp = self._initialize_experiment()
        exp.log_parameters(trial.params)
        exp.log_metrics(metrics)
        exp.add_tags([study.study_name])
        exp.log_other("trial_step", trial.number)
        exp.set_start_time(trial.datetime_start.timestamp())
        exp.set_end_time(trial.datetime_complete.timestamp())
        # wandb.config.update(attributes)
        # wandb.log({**trial.params, **metrics}, step=trial.number)

    def _initialize_experiment(self) -> comet_ml.APIExperiment:
        """Initializes Comet Experiment."""

        return comet_ml.APIExperiment(**self._comet_kwargs)
