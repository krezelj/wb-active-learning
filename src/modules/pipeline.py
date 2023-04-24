from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy
from torch.utils.data import DataLoader

import src.modules.data_module as dm
import src.modules.learner_module as lm


class Pipeline():
    
    __slots__ = ['data_set', 'learner', 'optimizer', 
                 'loss_function', 'n_queries', 'init_epochs', 
                 'epochs_per_query', 'query_batch_size']

    def __init__(self):
        self.data_set = None
        self.learner = None
        self.optimizer = None
        self.loss_function = None
        self.n_queries = None
        self.init_epochs = None
        self.epochs_per_query = None
        self.query_batch_size = None

    def fit(self, data_set:dm.ActiveDataset, learner:lm.ActiveLearner,
            optimizer, loss_function, n_queries, init_epochs, epochs_per_query,
            query_batch_size = 10):
        self.test_loader = DataLoader(data_set.test_set, batch_size=128)
        self.train_loader = DataLoader(data_set.labeled_set, batch_size=32, shuffle=True)
        self.data_set = data_set
        self.learner = learner
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.n_queries = n_queries
        self.init_epochs = init_epochs
        self.epochs_per_query = epochs_per_query
        self.query_batch_size = query_batch_size
    
    def transform(self):
        t_hist_all, v_hist_all = self.learner.fit(self.train_loader, self.test_loader,
                                                    self.optimizer, self.loss_function, 
                                                    early_stopping=True, epochs=self.init_epochs)
        a_hist_all = []
        a_hist_all.append(self._calculate_accuracy(self.test_loader).item())

        for i in range(self.n_queries):
            print(f"Iteration: {i+1}")
            
            # generate queries
            unlabeled_loader = DataLoader(self.data_set.unlabeled_set, batch_size=128, shuffle=False)
            queries, uncertainty = self.learner.generate_query(unlabeled_loader, criterion='entropy', batch_size=self.query_batch_size)
            self.data_set.get_label_by_idx(queries, move_sample=True)
            print(f"Max uncertainty: {uncertainty}")
            
            self.train_loader = DataLoader(self.data_set.labeled_set, batch_size=32, shuffle=True)

            t_hist, v_hist = self.learner.fit(self.train_loader, self.test_loader, self.optimizer,
                                               self.loss_function, epochs=self.epochs_per_query, 
                                               early_stopping=True, sample_weights=None)
                                        
            # save history
            t_hist_all.extend(t_hist)
            v_hist_all.extend(v_hist)
            a_hist_all.append(self._calculate_accuracy(self.learner, self.test_loader).item())
        #maybe better to save this arrays to .txt flat file cuz in evaluating pipeline in the loop will lost last arrays
        return t_hist_all, v_hist_all, a_hist_all

    def fit_transform(self, data_set:dm.ActiveDataset, learner:lm.ActiveLearner,
            optimizer, loss_function, n_queries, init_epochs, epochs_per_query,
            query_batch_size = 10):
            self.fit(data_set, learner, optimizer, loss_function, n_queries, init_epochs, 
                     epochs_per_query, query_batch_size)
            return self.transform()
            
    def _calculate_f1_score(self, test_loader):
        outputs = self.learner.predict(test_loader)
        return multiclass_f1_score(outputs, test_loader.dataset.dataset.targets[test_loader.dataset.indices])

    def _calculate_accuracy(self, test_loader):
        outputs = self.learner.predict(test_loader)
        return multiclass_accuracy(outputs, test_loader.dataset.dataset.targets[test_loader.dataset.indices])