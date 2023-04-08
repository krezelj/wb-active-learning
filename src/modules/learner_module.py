import torch.nn as nn


# TODO Move architectures to a separate file
class MNISTClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


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
            avg_vloss = self.__validate(validation_loader, loss_function)
            print(f"EPOCH {epoch+1}\n\tTraining: {avg_loss:.3f}\n\tValidation: {avg_vloss:.3f}")


    def __validate(self, validation_loader, loss_function):
        running_vloss = 0.0

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

    def generate_query(self, criterion='uncertainty'):
        raise NotImplementedError
    
