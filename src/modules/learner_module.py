import os

import torch.nn as nn
import torch
import numpy as np


class ActiveLearner():
    """
    Represents an active learner method.
    ...
    Attributes
    ----------
    model : torch.nn.Module
        The model for learning.
    device : str
        The device on which the learning takes place.

    Methods
    -------
    predict(data_loader)
        Makes predictions for the input data from a given DataLoader.
    validate(validation_loader, loss_function)
        Validates the model on the given DataLoader with a loss function.
    fit(training_loader, validation_loader, optimizer, loss_function, epochs=1, early_stopping=False, sample_weights=None, verbose=0, early_stopping_patience=5, early_stopping_threshold=0.001)
        Trains the model on the given training and validation data from a DataLoader.
    generate_query(unlabeled_loader, criterion='entropy', batch_size=1)
        Generates a query for sample indices, based on unlabeled data from a DataLoader.
    """

    
    __slots__ = ['model', 'device']

    def __init__(self, model, device) -> None:
        """
        Initializes an ActiveLearner object.

        Parameters
        ----------
        model : torch.nn.Module
            The model for learning.
        device : str
            The device on which the learning takes place.
        """

        self.model = model
        self.model.eval()
        self.device = device


    def __call__(self, inputs):
        """
        Parameters
        ----------
        inputs : torch.Tensor
            The input data.

        Returns
        -------
        torch.Tensor
            The model's output.
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
        return outputs
    
    def predict(self, data_loader): 
        """
        Makes predictions for the input data from a given DataLoader.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            The DataLoader containing the input data.

        Returns
        -------
        torch.Tensor
            The model's outputs for all input data.
        """

        all_outputs = []
        with torch.no_grad():
            for data in data_loader:
                inputs, *_ = data
                inputs = inputs.to(self.device)

                outputs = self(inputs)
                all_outputs.append(outputs)

        return torch.cat(all_outputs)

    def validate(self, validation_loader, loss_function):
        """
        Validates the model on the given DataLoader with a loss function.

        Parameters
        ----------
        validation_loader : torch.utils.data.DataLoader
            The DataLoader containing the validation data.
        loss_function : torch.nn.Module
            The loss function.

        Returns
        -------
        float
            The average loss value for the validation data.
        """

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
        """
        Trains the model on the given training and validation data from a DataLoader.

        Parameters
        ----------
        training_loader : torch.utils.data.DataLoader
            The DataLoader containing the training data.
        validation_loader : torch.utils.data.DataLoader
            The DataLoader containing the validation data.
        optimizer : torch.optim.Optimizer
            The optimizer for training the model.
        loss_function : torch.nn.Module
            The loss function.
        epochs : int, optional
            The number of training epochs. (default = 1)
        early_stopping : bool, optional
            Flag indicating whether to use early stopping. (default = False)
        sample_weights : torch.Tensor, optional
            Sample weights for weighted loss calculation. (default = None)
        verbose : int, optional
            Verbosity level for printing training progress. (default = 0)
            Options:
                0 - silent
                positive number - information about epoch number and average loss.
        early_stopping_patience : int, optional
            The number of epochs to wait for improvement before stopping early. (default = 5)
        early_stopping_threshold : float, optional
            The threshold for considering an improvement in validation loss. (default = 0.001)

        Returns
        -------
        list
            A list of average training loss values for each epoch.
        list
            A list of average validation loss values for each epoch.
        """

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
        """
        Trains the model for one epoch on the given training data.

        Parameters
        ----------
        training_loader : torch.utils.data.DataLoader
            The DataLoader containing the training data.
        loss_function : torch.nn.Module
            The loss function.
        optimizer : torch.optim.Optimizer
            The optimizer for training the model.
        sample_weights : torch.Tensor
            Sample weights for weighted loss calculation.

        Returns
        -------
        float
            The average training loss value for the epoch.
        """

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
        Generates a query for sample indices based on unlabeled data from a DataLoader.

        Parameters
        ----------
        unlabeled_loader : torch.utils.data.DataLoader
            The DataLoader containing the unlabeled data.
        criterion : str, optional
            The criterion for selecting samples in the query. (default = 'entropy')
            Options: 'entropy', 'margin', 'confidence', 'random'.
        batch_size : int, optional
            The number of samples to select in the query. (default = 1)

        Returns
        -------
        numpy.ndarray
            The indices of selected samples in the unlabeled data.
        torch.Tensor
            The uncertainties associated with the selected samples.
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
        """
        Calculates the entropy of predicted samples.

        Parameters
        ----------
        y_predicted : torch.Tensor
            The predicted probabilities.

        Returns
        -------
        torch.Tensor
            The calculated entropy values.
        """
        return -torch.sum(torch.mul(y_predicted, torch.log(torch.maximum(y_predicted, torch.tensor(1e-8)))), dim=1)

    @classmethod
    def __calculate_samples_margin(cls, y_predicted):
        """
        Calculates the margin of predicted samples.

        Parameters
        ----------
        y_predicted : torch.Tensor
            The predicted probabilities.

        Returns
        -------
        torch.Tensor
            The calculated margin values.
        """
        top2 = torch.topk(y_predicted, k=2, dim=1).values

        # return as (negative)margin so that argmin turns into argmax
        return -torch.abs(torch.diff(top2, dim=1)).reshape((-1,))

    @classmethod
    def __calculate_samples_confidence(cls, y_predicted):
        """
        Calculates the confidence of predicted samples.

        Parameters
        ----------
        y_predicted : torch.Tensor
            The predicted probabilities.

        Returns
        -------
        torch.Tensor
            The calculated confidence values.
        """
        return 1 - torch.max(y_predicted, dim=1).values
    
