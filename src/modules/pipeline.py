from __future__ import annotations
import os
import json
from typing import Any, Union, Literal

from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy
from torch.utils.data import DataLoader

import src.modules.data_module as dm
import src.modules.learner_module as lm
import src.modules.evaluation_module as em


class Pipeline:
    """
    A class used to initialize pipeline.
    ...
    Attributes
    ----------
    dataset: data_module.ActiveDataset
        A data set used to evaluation.
    learner: learner_module.ActiveLearner
        Type of method used to evaluate.
    settings: PipelineSettings
        Settings of pipeline.
    optimiser: torch.optim
        Type of used optimiser.
    loss_function: torch.nn.functional
        Type of loss function.
    train_loader: torch.utils.data.DataLoader
        Data set used for training.
    test_loader: torch.utils.data.DataLoader
        Data set used for testing.

    Methods
    -------
    run(verbose: int = 0, calculate_accuracy=False, calculate_f1_score=False)
         Runs the pipeline. Returns a session object and dict of stats such as loss history,
         accuracy and f1 score.
    """


    __slots__ = ['dataset', 'learner', 'settings', 'optimiser', 'loss_function',
                 'train_loader', 'test_loader']
    
    def __init__(self, dataset: dm.ActiveDataset, learner: lm.ActiveLearner,
                 optimiser, loss_function, settings: Union[dict, PipelineSettings]) -> None:
        """

        Parameters
        ----------
        dataset: data_module.ActiveDataset
            A data set used to evaluation.
        learner: learner_module.ActiveLearner
            Type of method used to evaluate.
        optimiser: torch.optim
            Type of used optimiser.
        loss_function: torch.nn.functional
            Type of loss function.
        settings: PipelineSettings
            Settings of pipeline.
        """

        self.dataset = dataset
        self.learner = learner
        self.optimiser = optimiser
        self.loss_function = loss_function

        self.test_loader = DataLoader(dataset.test_set, batch_size=128)
        self.train_loader = DataLoader(dataset.labeled_set, batch_size=32, shuffle=True)

        if type(settings) == dict:
            settings = PipelineSettings.from_dict(settings)
        self.settings: PipelineSettings = settings

    def run(self, verbose: int = 0,
            calculate_accuracy=False,
            calculate_f1_score=False) -> tuple[em.Session, dict[str, list[float]]]:
        """
        Runs the pipeline. Return a Session object as well as a dict of stats, which contains
        loss history on train & test and accuracy & f1 score history.

        Parameters
        ----------
            verbose: int, optional
                Which information should be displayed
                during session. (default=0)
                Options:
                    0 - silent
                    1 - display iteration count
                    2 - display max uncertainties in every iteration
            calculate_accuracy: bool, optional
                Whether or not to calculate predictions
                accuracy after every iteration.
            calculate_f1_score: bool, optional
                Whether or not to calculate predictions
                F1 score after every iteration.

        Pseudocode:

        perform initial learner fit
        loop for n_queries
              get_queries
              label_queries
              fit learner on updated train set
              update session
        return session
        """
        # initialise loss/accuracy history arrays
        # initial learner fit
        t_loss_history, v_loss_history = self.learner.fit(
            self.train_loader, self.test_loader,
            self.optimiser, self.loss_function,
            epochs=self.settings.init_epochs,
            early_stopping=True,
        )
        # initialise these anyway to keep the return type consistent:
        accuracy_history = []
        f1score_history = []

        if calculate_accuracy:
            accuracy_history.append(self._calculate_accuracy().item())
        if calculate_f1_score:
            f1score_history.append(self._calculate_f1_score().item())

        # initialise the session
        session = em.Session(dataset=self.dataset, learner=self.learner)

        for i in range(self.settings.n_queries):
            if verbose > 0:
                print(f"Iteration: {i+1}")

            # generate queries
            unlabeled_loader = DataLoader(self.dataset.unlabeled_set, batch_size=128, shuffle=False)
            queries, uncertainty = self.learner.generate_query(
                unlabeled_loader,
                criterion=self.settings.query_criterion,
                batch_size=self.settings.query_batch_size,
            )
            # label queries
            self.dataset.get_label_by_idx(queries, move_sample=True)
            if verbose > 1:
                print(f"Max uncertainty: {uncertainty}")
            self.train_loader = DataLoader(self.dataset.labeled_set, batch_size=32, shuffle=True)

            # fit learner on updated train set
            t_hist, v_hist = self.learner.fit(
                self.train_loader, self.test_loader,
                self.optimiser, self.loss_function,
                epochs=self.settings.epochs_per_query,
                early_stopping=True,
                sample_weights=None,
            )
            # update session
            session.update()

            # update loss/accuracy/f1score history
            t_loss_history.extend(t_hist)
            v_loss_history.extend(v_hist)
            if calculate_accuracy:
                accuracy_history.append(self._calculate_accuracy().item())
            if calculate_f1_score:
                f1score_history.append(self._calculate_f1_score().item())

        # pack loss/accuracy/f1score history arrays into one dict
        stats_to_return = {
            't_loss_history': t_loss_history,
            'v_loss_history': v_loss_history,
            'accuracy_history': accuracy_history,
            'f1score_history': f1score_history,
        }
        return session, stats_to_return

    def _calculate_f1_score(self):
        # Method used for calculating f1 score
        outputs = self.learner.predict(self.test_loader).to('cpu')
        return multiclass_f1_score(
            outputs,
            self.test_loader.dataset.dataset.targets[
                self.test_loader.dataset.indices]
        )

    def _calculate_accuracy(self):
        # Method used for calculating accuracy
        outputs = self.learner.predict(self.test_loader).to('cpu')
        return multiclass_accuracy(
            outputs,
            self.test_loader.dataset.dataset.targets[
                self.test_loader.dataset.indices]
        )

class PipelineSettings:

    """
    A class used to set pipeline settings.
    ...
    Attributes
    ----------
    n_queries: int
        Number of queries used in Active learning evaluation.
    init_epochs: int
        Number of initial epochs used in Active learning evaluation.
    epochs_per_query: int
        Number of epochs used for every query.
    query_batch_size: int
        Number of batches asked in every query.
    query_criterion: str
        Type of criterion used for selecting batches in query.

    Methods
    -------
    @staticmethod
    from_dict(settings_dict: dict[str, Any])
        Used for generating PiplineSettings from a dict.
    @staticmethod
    from_json(path_to_json: os.PathLike)
        Used for generating PiplineSettings from a JSON file.
    to_json()
        Used to export settings to a JSON file.
    to_dict()
        Used to export settings to a dict.
    """

    __slots__ = ['n_queries', 'init_epochs', 'epochs_per_query', 'query_batch_size',
                 'query_criterion']

    def __init__(self, n_queries: int, init_epochs: int,
                 epochs_per_query: int, query_batch_size: int = 100,
                 query_criterion: str = 'entropy') -> None:
        """

        Parameters
        ----------
        n_queries: int
            Number of queries used in Active learning evaluation.
        init_epochs: int
            Number of initial epochs used in Active learning evaluation.
        epochs_per_query: int
            Number of epochs used for every query.
        query_batch_size: int, optional
            Number of batches asked in every query. (default = 100)
        query_criterion: str, optional
            Type of criterion used for selecting batches in query. (default = 'entropy')
            Options: 'entropy' | 'margin' | 'confidence' | 'random'

        """
        self.n_queries = n_queries
        self.init_epochs = init_epochs
        self.epochs_per_query = epochs_per_query
        self.query_batch_size = query_batch_size
        self.query_criterion = query_criterion

    @staticmethod
    def from_dict(settings_dict: dict[str, Any]) -> PipelineSettings:
        """
        A named constructor which generates a PipelineSettings from a dict.
        Dict should be in a format like the one returned from `PipelineSettings.to_dict()` method.
        ...
        Parameters
        ----------
        settings_dict: dict
            Dictionary containing pipline settings.
        """
        try:
            settings_obj = PipelineSettings(
                n_queries=settings_dict['n_queries'],
                init_epochs=settings_dict['init_epochs'],
                epochs_per_query=settings_dict['epochs_per_query'],
                query_batch_size=settings_dict['query_batch_size'],
                query_criterion=settings_dict['query_criterion']
            )
            return settings_obj
        except KeyError:
            raise ValueError(f'Invalid dict format. Required keys: {PipelineSettings.__slots__}')

    def to_dict(self) -> dict[str, Any]:
        """
        Exports settings to a dict
        """
        return {
            'n_queries': self.n_queries,
            'init_epochs': self.init_epochs,
            'epochs_per_query': self.epochs_per_query,
            'query_batch_size': self.query_batch_size,
            'query_criterion': self.query_criterion,
        }

    @staticmethod
    def from_json(path_to_json: os.PathLike) -> PipelineSettings:
        """
        A named constructor which generates a PipelineSettings from a JSON file.
        A file should be in a format like the one generated by `PipelineSettings.to_json()` method.
        ...
        Parameters
        ----------
        path_to_json: os.PathLike
            Path to a JSON file from which we get settings.
        """
        with open(path_to_json, 'r') as file:
            settings_dict = json.load(file)
        return PipelineSettings.from_dict(settings_dict)

    def to_json(self, path: os.PathLike, filename: str = None) -> None:
        """
        Exports settings to a JSON file
        ...
        Parameters
        ----------
        path: os.PathLike
            Path where we want to save a JSON file.
        filename: str, optional
            Name of file we want to save. (default = None)
            If None filename is 'pipeline_settings.json'.

        """
        os.makedirs(path, exist_ok=True)
        # generic filename
        if filename is None:
            filename = 'pipeline_settings.json'
        # append extension to filename
        if not filename.endswith('.json'):
            filename += '.json'

        with open(os.path.join(path, filename), 'w') as file:
            json.dump(self.to_dict(), file)

    def __str__(self):
        return f'PipelineSettings({str(self.to_dict())})'

    def __repr__(self):
        return f'PipelineSettings({repr(self.to_dict())})'
