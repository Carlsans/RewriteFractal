import os
import random

import optuna
from typing import Callable, Dict, Any, Optional
from functools import partial

class StudyManager:
    def __init__(self,studyname,functiontooptimize):
        self.studyname = studyname
        self.storage = os.environ.get("GAOPTUNA_STORAGE", "sqlite:///gaoptuna.db")
        self.base_objective = functiontooptimize
        self.parameter_override = None
        self.function_modifier = None
    def starttrial(self,nostorage = False):
        print("Loading optuna study")
        if nostorage:
            study = optuna.create_study()
        else:
            study = optuna.load_study(
            study_name=self.studyname, storage=self.storage
            )
        study.optimize(self, n_trials=10000)
    def runUsingBestValuesAtRandom(self,k=15):
        bestparameters = self.getbestkparameters(k)
        print("Using existing study with best parameters :)")
        study = optuna.load_study(
            study_name="dummystudy", storage=self.storage
        )
        # study = optuna.create_study()
        while True :
            currentparam = random.choice(bestparameters)
            print("Current parameter is :",currentparam)
            self.set_parameters(currentparam)
            study.optimize(self, n_trials=1)
            # Delete the dummy study content to preserve memory
            # if random.random() > 0.95:
            #     self.clean_study("dummystudy")

    def clean_study(self,study_name: str):
        # Delete the existing study
        optuna.delete_study(study_name=study_name, storage=self.storage)

        # Create a fresh study
        optuna.create_study(study_name=study_name, storage=self.storage,sampler=optuna.samplers.NSGAIISampler)

    def flushdummystudy(self):
        optuna.delete_study(study_name="dummystudy", storage=self.storage)
        optuna.create_study(study_name="dummystudy", storage=self.storage,
                            sampler=optuna.samplers.NSGAIISampler)
        exit()
    def getbestkparameters(self,k):
        study = optuna.load_study(study_name=self.studyname, storage=self.storage)


        if k is not None:
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            sorted_trials = sorted(completed_trials,
                                   key=lambda t: t.value if t.value is not None else float('-inf'),
                                   reverse=(study.direction == optuna.study.StudyDirection.MAXIMIZE))

            results = []
            parameters = []
            for trial in sorted_trials[:k]:
                trial_info = {
                    'trial_id': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'datetime_start': trial.datetime_start,
                    'datetime_complete': trial.datetime_complete
                }
                results.append(trial_info)
                parameters.append(trial.params)
        print(parameters)
        del study
        return parameters
        # # Delete trials not in the keep set
        # for trial in trials:
        #     if trial.number not in trials_to_keep:
        #         study._storage.delete_trial(trial._trial_id)

        # for i, trial in enumerate(trials_to_keep):
        #     print(f"\nTrial {i + 1} (ID: {trial.number})")
        #     print(f"Value: {trial.value}")
        #     print("Parameters:")
        #     for param_name, param_value in trial.params.items():
        #         print(f"  {param_name}: {param_value}")

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Set specific parameter values to use instead of sampling.

        Args:
            params: Dictionary of parameter names and their values
        """
        self.parameter_override = params

    def clear_parameters(self) -> None:
        """Clear any parameter overrides."""
        self.parameter_override = None

    def modify_objective(self, modifier: Callable) -> None:
        """
        Set a function to modify the objective function's behavior.

        Args:
            modifier: Function that takes the original result and returns modified result
        """
        self.function_modifier = modifier

    def clear_modifier(self) -> None:
        """Clear any objective function modifications."""
        self.function_modifier = None

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Run the trial with current settings.

        Args:
            trial: Optuna trial object

        Returns:
            float: The objective value
        """
        # If parameters are overridden, suggest them directly
        if self.parameter_override is not None:
            for param_name, param_value in self.parameter_override.items():
                if isinstance(param_value, int):
                    trial.suggest_int(param_name, param_value, param_value)
                elif isinstance(param_value, float):
                    trial.suggest_float(param_name, param_value, param_value)
                else:
                    trial.suggest_categorical(param_name, [param_value])

        # Run the base objective
        result = self.base_objective(trial)

        # Apply modifier if set
        if self.function_modifier is not None:
            result = self.function_modifier(result)

        return result