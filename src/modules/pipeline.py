from __future__ import annotations
import os
import json
from typing import Any, Union

from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy
from torch.utils.data import DataLoader

import src.modules.data_module as dm
import src.modules.learner_module as lm
import src.modules.evaluation_module as em


# class Pipeline():
    
#     __slots__ = ['data_set', 'learner', 'optimizer', 
#                  'loss_function', 'n_queries', 'init_epochs', 
#                  'epochs_per_query', 'query_batch_size', 'test_loader', 'train_loader']

#     def __init__(self):
#         self.data_set = None
#         self.learner = None
#         self.optimizer = None
#         self.loss_function = None
#         self.n_queries = None
#         self.init_epochs = None
#         self.epochs_per_query = None
#         self.query_batch_size = None

#     def fit(self, data_set:dm.ActiveDataset, learner:lm.ActiveLearner,
#             optimizer, loss_function, n_queries, init_epochs, epochs_per_query,
#             query_batch_size = 10):
#         self.test_loader = DataLoader(data_set.test_set, batch_size=128)
#         self.train_loader = DataLoader(data_set.labeled_set, batch_size=32, shuffle=True)
#         self.data_set = data_set
#         self.learner = learner
#         self.optimizer = optimizer
#         self.loss_function = loss_function
#         self.n_queries = n_queries
#         self.init_epochs = init_epochs
#         self.epochs_per_query = epochs_per_query
#         self.query_batch_size = query_batch_size
    
#     def transform(self):
#         t_hist_all, v_hist_all = self.learner.fit(self.train_loader, self.test_loader,
#                                                     self.optimizer, self.loss_function, 
#                                                     early_stopping=True, epochs=self.init_epochs)
#         a_hist_all = []
#         a_hist_all.append(self._calculate_accuracy(self.test_loader).item())

#         for i in range(self.n_queries):
#             print(f"Iteration: {i+1}")
            
#             # generate queries
#             unlabeled_loader = DataLoader(self.data_set.unlabeled_set, batch_size=128, shuffle=False)
#             queries, uncertainty = self.learner.generate_query(unlabeled_loader, criterion='entropy', batch_size=self.query_batch_size)
#             self.data_set.get_label_by_idx(queries, move_sample=True)
#             print(f"Max uncertainty: {uncertainty}")
            
#             self.train_loader = DataLoader(self.data_set.labeled_set, batch_size=32, shuffle=True)

#             t_hist, v_hist = self.learner.fit(self.train_loader, self.test_loader, self.optimizer,
#                                                self.loss_function, epochs=self.epochs_per_query, 
#                                                early_stopping=True, sample_weights=None)
                                        
#             # save history
#             t_hist_all.extend(t_hist)
#             v_hist_all.extend(v_hist)
#             a_hist_all.append(self._calculate_accuracy(self.learner, self.test_loader).item())
#         #maybe better to save this arrays to .txt flat file cuz in evaluating pipeline in the loop will lost last arrays
#         return t_hist_all, v_hist_all, a_hist_all

#     def fit_transform(self, data_set:dm.ActiveDataset, learner:lm.ActiveLearner,
#             optimizer, loss_function, n_queries, init_epochs, epochs_per_query,
#             query_batch_size = 10):
#             self.fit(data_set, learner, optimizer, loss_function, n_queries, init_epochs, 
#                      epochs_per_query, query_batch_size)
#             return self.transform()
            
#     def _calculate_f1_score(self, test_loader):
#         outputs = self.learner.predict(test_loader)
#         return multiclass_f1_score(outputs, test_loader.dataset.dataset.targets[test_loader.dataset.indices])

#     def _calculate_accuracy(self, test_loader):
#         outputs = self.learner.predict(test_loader)
#         return multiclass_accuracy(outputs, test_loader.dataset.dataset.targets[test_loader.dataset.indices])
    

class Pipeline():
     
    __slots__ = ['ds', 'learner', 'settings', 'optimiser', 'loss_function']
    
    def __init__(self, ds, learner, optimiser, loss_function,
                 settings: Union[dict, PipelineSettings]) -> None:
        self.ds = ds
        self.learner = learner
        self.optimiser = optimiser
        self.loss_function = loss_function

        if type(settings) == dict:
            settings = PipelineSettings.from_dict(settings)
        self.settings: PipelineSettings = settings

    def run(self) -> em.Session:
        # perform initial learner fit
        # loop for n_queries
        #       get_queries
        #       label_queries
        #       fit learner on updated train set
        #       update session
        # return session
        pass


class PipelineSettings:

    __slots__ = ['n_queries', 'init_epochs', 'epochs_per_query', 'query_batch_size']

    def __init__(self, n_queries, init_epochs, epochs_per_query, query_batch_size):
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
