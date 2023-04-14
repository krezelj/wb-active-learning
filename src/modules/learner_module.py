import torch.nn as nn
import torch


class ActiveLearner():
    
    __slots__ = ['model', 'device']

    def __init__(self, model, device):
        self.model = model
        self.device = device


    def __call__(self, inputs):
        return self.model(inputs)
    

    def fit(self, training_loader, validation_loader, optimizer, loss_function, epochs=1):
        for epoch in range(epochs):

            # train model
            self.model.train(True)
            avg_loss = self.__train_one_epoch(training_loader, loss_function, optimizer)
            self.model.train(False)

            # validate model
            avg_vloss = self.validate(validation_loader, loss_function)
            print(f"EPOCH {epoch+1}\n\tTraining: {avg_loss:.3f}\n\tValidation: {avg_vloss:.3f}")


    def validate(self, validation_loader, loss_function):
        running_vloss = 0.0

        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(self.device)
                vlabels = vlabels.to(self.device)

                voutputs = self.model(vinputs)
                vloss = loss_function(voutputs, vlabels)
                running_vloss += vloss

        return running_vloss / (i + 1)

    def __train_one_epoch(self, training_loader, loss_function, optimizer):
        running_loss = 0.0

        for i, data in enumerate(training_loader):
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = loss_function(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        return running_loss / (i + 1)

    def predict(self):
        # TODO method that takes in dataset/dataloader (?)
        # and outputs predictions for all observations

        # TODO after implementing use it in generate_query method 
        # (=> change generate_query to accept dataset/dataloder accordingly)
        
        raise NotImplementedError


    def generate_query(self, unlabeled_loader, criterion='uncertainty'):
        # For now only uncertainty is available and uses *entropy* as measure of uncertainty
        all_outputs = []
        with torch.no_grad():
            for i, udata in enumerate(unlabeled_loader):
                uinputs, _ = udata
                uinputs = uinputs.to(self.device)

                uoutputs = self.model(uinputs)
                all_outputs.append(uoutputs)

        y_pred = torch.cat(all_outputs)
        y_pred = nn.functional.softmax(y_pred, dim=1)
        uncertainties = -torch.sum(torch.mul(y_pred, torch.log(y_pred)), dim=1) # entropy

        # most uncertain sample, convert to cpu for easy indexing as DataSet works on a cpu
        query = torch.argmax(uncertainties).to('cpu') 
        return query
    
