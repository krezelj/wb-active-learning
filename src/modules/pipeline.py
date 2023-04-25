from __future__ import annotations
import os
import json
from typing import Any, Union

from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy
from torch.utils.data import DataLoader

import src.modules.data_module as dm
import src.modules.learner_module as lm
import src.modules.evaluation_module as em


class Pipeline:
     
    __slots__ = ['dataset', 'learner', 'settings', 'optimiser', 'loss_function',
                 'train_loader', 'test_loader']
    
    def __init__(self, dataset: dm.ActiveDataset, learner: lm.ActiveLearner,
                 optimiser, loss_function, settings: Union[dict, PipelineSettings]):
        self.dataset = dataset
        self.learner = learner
        self.optimiser = optimiser
        self.loss_function = loss_function

        self.test_loader = DataLoader(dataset.test_set, batch_size=128)
        self.train_loader = DataLoader(dataset.labeled_set, batch_size=32, shuffle=True)

        if type(settings) == dict:
            settings = PipelineSettings.from_dict(settings)
        self.settings: PipelineSettings = settings

    def run(self) -> tuple[em.Session, dict[str, list[float]]]:
        """
        Runs the pipeline. Return a Session object as well as a dict of stats, which contains
        loss history on train and test and accuracy history

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
        accuracy_history = [self._calculate_accuracy(self.test_loader).item()]

        # initialise the session
        session = em.Session(dataset=self.dataset, learner=self.learner)

        for i in range(self.settings.n_queries):
            print(f"Iteration: {i+1}")

            # generate queries
            unlabeled_loader = DataLoader(self.dataset.unlabeled_set, batch_size=128, shuffle=False)
            queries, uncertainty = self.learner.generate_query(
                unlabeled_loader,
                criterion='entropy',
                batch_size=self.settings.query_batch_size,
            )
            # label queries
            self.dataset.get_label_by_idx(queries, move_sample=True)
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

            # update loss/accuracy history
            t_loss_history.extend(t_hist)
            v_loss_history.extend(v_hist)
            accuracy_history.append(self._calculate_accuracy(self.test_loader).item())

        # pack loss/accuracy history arrays into one dict
        stats_to_return = {
            't_loss_history': t_loss_history,
            'v_loss_history': v_loss_history,
            'accuracy_history': accuracy_history,
        }
        return session, stats_to_return

    def _calculate_f1_score(self, test_loader):
        outputs = self.learner.predict(test_loader)
        return multiclass_f1_score(
            outputs,
            test_loader.dataset.dataset.targets[test_loader.dataset.indices]
        )

    def _calculate_accuracy(self, test_loader):
        outputs = self.learner.predict(test_loader)
        return multiclass_accuracy(
            outputs,
            test_loader.dataset.dataset.targets[test_loader.dataset.indices]
        )


class PipelineSettings:

    __slots__ = ['n_queries', 'init_epochs', 'epochs_per_query', 'query_batch_size']

    def __init__(self, n_queries: int, init_epochs: int,
                 epochs_per_query: int, query_batch_size: int = 100):
        self.n_queries = n_queries
        self.init_epochs = init_epochs
        self.epochs_per_query = epochs_per_query
        self.query_batch_size = query_batch_size

    @staticmethod
    def from_dict(settings_dict: dict[str, Any]) -> PipelineSettings:
        """
        A named constructor which generates a PipelineSettings from a dict.
        Dict should be in a format like the one returned from `PipelineSettings.to_dict()` method.
        """
        try:
            settings_obj = PipelineSettings(
                n_queries=settings_dict['n_queries'],
                init_epochs=settings_dict['init_epochs'],
                epochs_per_query=settings_dict['epochs_per_query'],
                query_batch_size=settings_dict['query_batch_size'],
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
        }

    @staticmethod
    def from_json(path_to_json: os.PathLike) -> PipelineSettings:
        """
        A named constructor which generates a PipelineSettings from a JSON file.
        A file should be in a format like the one generated by `PipelineSettings.to_json()` method.
        """
        with open(path_to_json, 'r') as file:
            settings_dict = json.load(file)
        return PipelineSettings.from_dict(settings_dict)

    def to_json(self, path: os.PathLike, filename: str = None):
        """
        Exports settings to a JSON file
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
