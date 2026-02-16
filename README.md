# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="1209" height="799" alt="image" src="https://github.com/user-attachments/assets/2a210679-0a23-4590-9a97-84b5380f5907" />


## DESIGN STEPS

### STEP 1: Data Collection and Understanding
Collect customer data from the existing market and identify the features that influence customer segmentation. Define the target variable as the customer segment (A, B, C, or D).

### STEP 2: Data Preprocessing
Remove irrelevant attributes, handle missing values, and encode categorical variables into numerical form. Split the dataset into training and testing sets.

### STEP 3: Model Design and Training
Design a neural network classification model with suitable input, hidden, and output layers. Train the model using the training data to learn patterns for customer segmentation.

### STEP 4: Model Evaluation and Prediction
Evaluate the trained model using test data and use it to predict the customer segment for new customers in the target market.


## PROGRAM

### Name: Kevin P
### Register Number: 212224040159

```python
# Define Neural Network(Model1)
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

```

```python
# Initialize the Model, Loss Function, and Optimizer
model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
train_model(model, train_loader, criterion, optimizer, epochs=100)

```

```python
#function to train the model
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
          optimizer.zero_grad()
          outputs=model(inputs)
          loss=criterion(outputs, labels)
          loss.backward()
          optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```

## Dataset Information

<img width="1154" height="243" alt="image" src="https://github.com/user-attachments/assets/a0ea83a6-2651-4421-9785-533da43a54ce" />


## OUTPUT

### Confusion Matrix

<img width="372" height="114" alt="image" src="https://github.com/user-attachments/assets/93d4d4a5-d6fd-469f-a4b0-0930ca71dad4" />

<img width="866" height="648" alt="image" src="https://github.com/user-attachments/assets/ed2ce27d-c1e7-4814-a51d-cdc97c441399" />


### Classification Report

<img width="1021" height="245" alt="image" src="https://github.com/user-attachments/assets/d0d7be07-06bc-4393-a206-08b5ef3263a5" />



### New Sample Data Prediction

<img width="1671" height="103" alt="image" src="https://github.com/user-attachments/assets/57d09063-1447-4a9e-a527-71de4a92aea3" />



## RESULT

Thus neural network classification model is developded for the given dataset. 
