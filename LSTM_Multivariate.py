import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from deap import base, creator, tools, algorithms

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Specify the device (cuda:0 for GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### Data preparation

# Load the stock market data
df = pd.read_csv('./data/MSFT.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)


def prepare_data(data, look_back, index_X, index_Y):
    X, y = [], []
    for i in range(len(data) - look_back - 1):
        X.append(data[i:(i + look_back), index_X])
        y.append(data[i + look_back, index_Y])
    return torch.tensor(np.array(X)), torch.tensor(np.array(y))

# Preprocess the data
look_back = 14  # Choose the number of previous time steps to use as input
INDEX_Y = 0  # Choose the column to use as a target
INDEX_X = 0  # Column for analysis

scaler = preprocessing.StandardScaler()  ##MinMaxScaler(feature_range=(0, 1))
data = df.values
data_scaled = scaler.fit_transform(data)

X, y = prepare_data(data_scaled, look_back, index_X=INDEX_X, index_Y=INDEX_Y)

if len(X.shape) == 2:
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
else:
    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
input_shape = (X.shape[1], X.shape[2])


# Move existing arrays to the GPU
X, y = map(lambda x: x.clone().detach().requires_grad_(True), (torch.tensor(X), torch.tensor(y)))
X, y = X.float().to(device), y.float().to(device)

tscv = TimeSeriesSplit(n_splits=2)  # Choose the number of splits

for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


def create_dataset(dataset, time_steps=1):
    X, Y = [], []
    for i in range(len(dataset) - time_steps - 1):
        a = dataset[i:(i + time_steps), 0]
        X.append(a)
        Y.append(dataset[i + time_steps, 0])

    return np.array(X), np.array(Y)


# Function to return the unscaled predicted prices
def unscal_data(data, scaler, index):
    array = np.zeros(shape=(len(data), scaler.n_features_in_)) #Pour MinMaxScaler
    array[:, index] = data.ravel()
    unscaled_data = scaler.inverse_transform(array)[:, index]

    return unscaled_data


# Create LSTM model
class LSTMModel(nn.Module):
    def __init__(self, dropout_rate, hidden_size, num_layers, input_size=1, output_size=1):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size

        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_size, hidden_size=self.hidden_size, num_layers=num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_2 = nn.Linear(num_layers * hidden_size, output_size)

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]

        # layer 1
        x = self.linear_1(x)
        x = self.relu(x)

        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)

        # reshape output from hidden cell into [batch, features] for `linear_2`
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

        # layer 2
        x = self.dropout(x)
        predictions = self.linear_2(x)
        return predictions[:, -1]


def train_model(model, X, y, optimizer, learning_rate, epochs):

    optimizer_class = getattr(torch.optim, optimizer)
    optimizer2 = optimizer_class(model.parameters(), lr=learning_rate)

    #Defining the Loss function
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(X, y), shuffle=False, batch_size=16)

    model.train()

    prev_loss = float('inf')  # Initialize previous loss
    unchanged_count = 0  # Initialize unchanged count

    for epoch in range(epochs):
        total_loss = 0.0

        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            total_loss += loss.item()  # Accumulate the loss

            optimizer2.zero_grad()

            loss.backward(retain_graph=True)
            optimizer2.step()

        epoch_loss = total_loss / len(loader)
        print(f'Epoch {epoch + 1} Loss: {epoch_loss}')

        # Check if loss hasn't changed by more than 0.005 for last 30 iterations
        if abs(epoch_loss - prev_loss) < 0.01:
            unchanged_count += 1
            if unchanged_count >= 50:
                print("Loss hasn't changed for the last XX iterations. Stopping training.")
                break
        else:
            unchanged_count = 0

        prev_loss = epoch_loss

    return 1/(1+epoch_loss)


# Define the evaluation function that will be optimized by the GA
def evaluate_lstm(individual, X, y, input_shape, epoch):

    units, dropout_rate, optimizer, learning_rate, num_layers = individual[:5]
    model = LSTMModel(hidden_size=int(units), dropout_rate=dropout_rate, num_layers=int(num_layers)).to(device)
    mean_loss = train_model(model=model, X=X, y=y, optimizer=optimizer, epochs=epoch, learning_rate=learning_rate)

    return mean_loss,

# Define the custom mutation function
def custom_mutation(individual, indpb_units=0.1, indpb_dropout_rate=0.1, indpb_optimizer=0.1, indpb_learning_rate=0.1,
                    indpb_num_layers=0.1):
    if random.random() < indpb_units:
        individual[0] = toolbox.units()
    if random.random() < indpb_dropout_rate:
        individual[1] = toolbox.dropout_rate()
    if random.random() < indpb_optimizer:
        individual[2] = toolbox.optimizer()
    if random.random() < indpb_learning_rate:
        individual[3] = toolbox.learning_rate()
    if random.random() < indpb_num_layers:
        individual[4] = toolbox.num_layers()
    return individual,


def store_fitness_values(population, generation, fitness_values, generations, elapsed_times):
    for ind in population:
        fitness_values.append(ind.fitness.values[0])
        generations.append(generation)
        elapsed_times.append(time.time())


def eaSimpleWithFitnessStorage(population, toolbox, cxpb, mutpb, ngen, stats=None,
                               halloffame=None, verbose=__debug__, fitness_values=None, generations=None,
                               elapsed_times=None):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the old population by the offspring
        population[:] = offspring

        # Store the fitness values and generations
        if fitness_values is not None and generations is not None:
            store_fitness_values(population, gen, fitness_values, generations, elapsed_times)

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def plot_predictions(test_dates, actual_prices, predicted_prices):
    plt.figure(figsize=(16, 8))
    plt.plot(test_dates, actual_prices, color='blue', label='Actual Prices')
    plt.plot(test_dates, predicted_prices, color='red', label='Predicted Prices')
    plt.xlabel('Dates')
    plt.ylabel('Prices')
    plt.title('Stock Market Price Prediction using LSTM and Genetic Algorithm')
    plt.legend()
    plt.show()


### Set up the genetic algorithm ####
creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register('units', random.uniform, 5, 20)
toolbox.register('dropout_rate', random.uniform, 0.3, 0.7)
toolbox.register('optimizer', random.choice,
                 ['Adagrad', 'Adam', 'Adamax', 'RMSprop', 'SGD'])  # ['Adagrad', 'Adam', 'Adamax', 'RMSprop', 'SGD']
toolbox.register('learning_rate', random.uniform, 0.000001, 0.001)
toolbox.register('num_layers', random.uniform, 1, 2)

toolbox.register('individual', tools.initCycle, creator.Individual,
                 (toolbox.units, toolbox.dropout_rate, toolbox.optimizer, toolbox.learning_rate, toolbox.num_layers), n=1)

toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('mate', tools.cxOnePoint)

toolbox.register('mutate', custom_mutation,
                 indpb_units=0.2, indpb_dropout_rate=0.2, indpb_optimizer=0.2, indpb_learning_rate=0.2)

toolbox.register('select', tools.selBest)
toolbox.register('evaluate', evaluate_lstm, X=X_train,
                 y=y_train, input_shape=input_shape, epoch=3)

population = toolbox.population(n=4)  # Number of individuals
ngen = 20  # Number of generations

fitness_values = []
generations = []
elapsed_times = []

result, log = eaSimpleWithFitnessStorage(
    population, toolbox, cxpb=0.5, mutpb=0.3, ngen=ngen, verbose=True,
    fitness_values=fitness_values, generations=generations, elapsed_times=elapsed_times)

elapsed_times = [t for t in elapsed_times]

### Once optimize, create a load the LSTM model with the optimized parameters
best_individual = tools.selBest(population, 1)[0]
units = best_individual[0]
dropout_rate = best_individual[1]
optimizer = best_individual[2]
learning_rate = best_individual[3]
num_layers = best_individual[4] #Number of LSTM layers

best_model = LSTMModel(hidden_size=int(units), dropout_rate=dropout_rate, num_layers=int(num_layers)).to(device)

train_model(model=best_model, X=X_train, y=y_train, optimizer=optimizer, learning_rate=learning_rate, epochs=300)

def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        pred = model(data)

    return pred


# Function to calculate when the algorithm predicted the movement correctly
def calculate_variation(prices):
    return np.where(np.diff(prices) >= 0, 1, 0)

y_pred = evaluate(model=best_model, data=X_test)

y_predicted = unscal_data(data=y_pred.cpu(), scaler=scaler, index=INDEX_Y)
y_train = unscal_data(data=y_train.detach().cpu().numpy(), scaler=scaler, index=INDEX_Y)
y_test = unscal_data(data=y_test.detach().cpu().numpy(), scaler=scaler, index=INDEX_Y)

plot_predictions(test_dates=df.index[test_index], actual_prices=y_test,
                 predicted_prices=y_predicted)

actual_variation = calculate_variation(y_test)
predicted_variation = calculate_variation(y_predicted)
print(np.mean(predicted_variation == actual_variation))

mse_test = mean_squared_error(y_test, y_predicted)
mae_test = mean_absolute_error(y_test, y_predicted)
mape_test = np.mean(np.abs((y_test, y_predicted) / y_test)) * 100

print(f'MSE : {mse_test}')
print(f'MAE : {mae_test}')
print(f'MAPE : {mape_test}')
print(best_individual)
