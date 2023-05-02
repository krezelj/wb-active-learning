import os

import torch.nn as nn
import torch
import numpy as np


class ActiveLearner():
    
    __slots__ = ['model', 'device']

    def __init__(self, model, device):
        self.model = model
        self.model.eval()
        self.device = device


    def __call__(self, inputs):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
        return outputs
    
    def predict(self, data_loader):        
        all_outputs = []
        with torch.no_grad():
            for data in data_loader:
                inputs, *_ = data
                inputs = inputs.to(self.device)

                outputs = self(inputs)
                all_outputs.append(outputs)

        return torch.cat(all_outputs)

    def validate(self, validation_loader, loss_function):
        if len(validation_loader) == 0:
            print("Warning! Empty validation_loader")
            return np.nan
        
        running_vloss = 0.0
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels, *_ = vdata
                vinputs = vinputs.to(self.device)
                vlabels = vlabels.to(self.device)

                voutputs = self(vinputs)
                vloss = loss_function(voutputs, vlabels)
                running_vloss += vloss.item()

        return running_vloss / (i + 1)

    def fit(self, training_loader, validation_loader, 
            optimizer, loss_function, 
            epochs=1, 
            early_stopping=False,
            sample_weights=None,
            verbose=0,
            early_stopping_patience = 5,
            early_stopping_threshold = 0.001):

        avg_loss_history = []
        avg_vloss_history = []

        min_avg_vloss = np.inf
        early_stopping_counter = 0


        # create a temporary directory for model states if it doesn't exist yet
        model_state_dir = './temp_model_states'
        os.makedirs(model_state_dir, exist_ok=True)

        for epoch in range(epochs):

            # train model
            self.model.train(True)
            avg_loss = self.__train_one_epoch(training_loader, loss_function, optimizer, sample_weights)
            avg_loss_history.append(avg_loss)
            self.model.train(False)

            # validate model
            if validation_loader is not None:
                avg_vloss = self.validate(validation_loader, loss_function)
                avg_vloss_history.append(avg_vloss)
            else:
                avg_vloss = np.nan

            
            # print info
            if verbose > 0:
                print(f"EPOCH {epoch+1}\n\tTraining: {avg_loss:.3f}\n\tValidation: {avg_vloss:.3f}")

            if avg_vloss < min_avg_vloss:
                early_stopping_counter = 0
                torch.save(self.model.state_dict(), f'{model_state_dir}/best_model.pth')
                min_avg_vloss = avg_vloss
            elif avg_vloss > min_avg_vloss + early_stopping_threshold:
                early_stopping_counter += 1
            if early_stopping:
                if early_stopping_counter == early_stopping_patience:
                    if verbose > 0: 
                        print("Stopping Early...")
                    self.model.load_state_dict(torch.load(f'{model_state_dir}/best_model.pth'))
                    return avg_loss_history, avg_vloss_history

        # TODO Add a flag whether the best model should be retrived or not
        # (maybe the user wants the laters model to be present)
        self.model.load_state_dict(torch.load(f'{model_state_dir}/best_model.pth'))
        return avg_loss_history, avg_vloss_history

    def __train_one_epoch(self, training_loader, loss_function, optimizer, sample_weights):
        running_loss = 0.0

        for i, data in enumerate(training_loader):
            inputs, labels, *idx = data
            if len(idx) > 0: 
                idx = idx[0]

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(inputs)

            if sample_weights is None:
                loss = loss_function(outputs, labels)
            else:
                per_sample_loss = loss_function(outputs, labels, reduction='none')
                batch_sample_weights = sample_weights[idx]
                weighted_loss = per_sample_loss * batch_sample_weights.to(self.device)
                loss = torch.mean(weighted_loss)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        return running_loss / (i + 1)

    def generate_query(self, unlabeled_loader, criterion='entropy', batch_size=1):
        """
        criterion: 'entropy' | 'margin' | 'confidence' | 'random' |
        """

        if criterion == 'random':
            return np.random.choice(np.arange(len(unlabeled_loader.dataset)), size=batch_size, replace=False), None

        y_predicted = self.predict(unlabeled_loader)

        # TODO right now this is tied to MNISTClassifier model that outputs logits
        # ideally it should be output-aware and not apply softmax to every output
        y_predicted = nn.functional.softmax(y_predicted, dim=1) 

        if criterion == 'entropy':
            uncertainties = ActiveLearner.__calculate_samples_entropy(y_predicted)
        elif criterion == 'margin':
            uncertainties = ActiveLearner.__calculate_samples_margin(y_predicted)
        elif criterion == 'confidence':
            uncertainties = ActiveLearner.__calculate_samples_confidence(y_predicted)    
        else:
            raise ValueError("Invalid criterion. Must be one of `entropy`, `margin`, `confidence`")
        

        # myopical batch-mode most queries
        queries = torch.topk(uncertainties.flatten(), batch_size).indices
        
        # TODO return predictions as well (predictions before applying softmax, i.e. raw predictions)
        return queries.to('cpu'), uncertainties[queries]
    

    # TODO Move this classes to utilities (?)
    # I'm not sure if they belong here
    @classmethod
    def __calculate_samples_entropy(cls, y_predicted):
        return -torch.sum(torch.mul(y_predicted, torch.log(torch.maximum(y_predicted, torch.tensor(1e-8)))), dim=1)

    @classmethod
    def __calculate_samples_margin(cls, y_predicted):
        top2 = torch.topk(y_predicted, k=2, dim=1).values

        # return as (negative)margin so that argmin turns into argmax
        return -torch.abs(torch.diff(top2, dim=1)).reshape((-1,))

    @classmethod
    def __calculate_samples_confidence(cls, y_predicted):
        return 1 - torch.max(y_predicted, dim=1).values
    
