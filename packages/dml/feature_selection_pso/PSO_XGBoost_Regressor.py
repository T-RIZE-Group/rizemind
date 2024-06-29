import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

class PSO_XGBoost_regressor():

    def __init__(self, data, target, popSize, maxIt):
        self.popSize = popSize
        self.maxIt = maxIt
        self.x_data = data
        self.y_data = target
        self.features = self.x_data.columns.tolist()
        self.dim = self.x_data.shape[1]
        self.testSize = 0.3
        self.bestModel = None
        self.gbestPos = None
        self.gbestCost = float("inf")

        # Drop 'Unnamed: 0' column if it exists
        if 'Unnamed: 0' in self.x_data.columns:
            self.x_data = self.x_data.drop(columns=['Unnamed: 0'])

        # Drop 'price_label' and 'price' columns if they exist
        columns_to_drop = ['price_label', 'price']
        existing_columns = [col for col in columns_to_drop if col in self.x_data.columns]
        if existing_columns:
            self.x_data = self.x_data.drop(columns=existing_columns)

    def fit(self, isPlot=False):
        c1 = 2
        c2 = 2
        w = 1
        wDamp = 0.99
        population, velocity, cost = self.initialization()
        pbestPos = population.copy()
        pbestCost = cost.copy()
        gbestCost_it = []
        gbestAverageCost_it = []
        gbestCost_it.append(self.gbestCost)
        gbestAverageCost_it.append(np.mean(cost))

        ##### main loop
        for it in range(self.maxIt):
            for p in range(len(population)):
                particle = population[p]
                velocity[p] = w * velocity[p] + c1 * np.random.rand() * (pbestPos[p] - particle) + c2 * np.random.rand() * (self.gbestPos - particle)
                particle = particle + velocity[p]
                particle[particle >= 0.5] = 1
                particle[particle < 0.5] = 0
                if np.all(particle == 0):
                    random_index = np.random.randint(0, len(particle))
                    particle[random_index] = 1
                population[p] = particle
                cost[p], trainedModel = self.costEval(particle)

                if cost[p] < self.gbestCost:
                    self.gbestCost = cost[p]
                    self.gbestPos = population[p]
                    self.bestModel = trainedModel

                if cost[p] < pbestCost[p]:
                    pbestCost[p] = cost[p]
                    pbestPos[p] = population[p]

            w = wDamp * w
            gbestCost_it.append(self.gbestCost)
            gbestAverageCost_it.append(np.average(cost))
            print('iteration: ', it, ' best cost: ', self.gbestCost, ' average cost: ', gbestAverageCost_it[it])

        if isPlot:
            iterations = [iter for iter in range(1, self.maxIt + 2)]
            plt.plot(iterations, gbestCost_it, label="best cost")
            plt.plot(iterations, gbestAverageCost_it, label="average cost")
            plt.legend()
            plt.xlabel('Iterations')
            plt.ylabel('Cost')
            plt.title('Cost vs Iterations')
            plt.show()

        # Ensure gbestPos has correct dimensions
        if len(self.gbestPos) > self.x_data.shape[1]:
            self.gbestPos = self.gbestPos[:self.x_data.shape[1]]

        # Get the names of top 4 features
        feature_names = self.x_data.columns[np.where(self.gbestPos == 1)[0]]
        top_features = feature_names[:4]
        print(f"Top 4 selected features: {', '.join(top_features)}")

    def initialization(self):
        population = []
        velocity = []
        cost = []
        trainedModels = []
        for i in range(self.popSize):
            random_particle = np.array([random.randint(0, 1) for _ in range(self.dim)])
            if np.all(random_particle == 0):
                random_index = np.random.randint(0, len(random_particle))
                random_particle[random_index] = 1

            population.append(random_particle)
            velocity.append(np.array([random.randint(0, 0) for _ in range(self.dim)]))
            costval, regModel = self.costEval(population[i])
            cost.append(costval)
            if cost[i] < self.gbestCost:
                self.gbestPos = population[i]
                self.gbestCost = cost[i]
                self.bestModel = regModel
        return population, velocity, cost

    def costEval(self, particle):
        zero_indices = np.where(particle == 0)[0]
        dataset = self.x_data.copy()

        # Drop non-selected columns based on indices
        selected_columns = [self.features[idx] for idx in np.where(particle == 1)[0]]
        dataset = dataset[selected_columns]

        # Encode categorical columns
        categorical_columns = dataset.select_dtypes(include=['object']).columns.tolist()
        dataset = self.encode_categorical(dataset, categorical_columns)

        target = self.y_data.copy()
        x_train, x_test, y_train, y_test = train_test_split(dataset, target, test_size=self.testSize, random_state=42)

        model = xgb.XGBRegressor(objective='reg:squarederror')
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        costVal = mean_squared_error(y_test, y_pred)
        return costVal, model

    def encode_categorical(self, dataset, categorical_columns):
        label_encoders = {}
        for col in categorical_columns:
            if col in dataset.columns:
                le = LabelEncoder()
                dataset[col] = le.fit_transform(dataset[col])
                label_encoders[col] = le
            else:
                print(f"Column {col} not found in the dataset")
        return dataset

    def predict(self, x_test):
        zero_indices = np.where(self.gbestPos == 0)[0]
        dataset = x_test.copy()

        # Drop non-selected columns based on indices
        selected_columns = [self.features[idx] for idx in np.where(self.gbestPos == 1)[0]]
        dataset = dataset[selected_columns]

        # Encode categorical columns
        categorical_columns = dataset.select_dtypes(include=['object']).columns.tolist()
        dataset = self.encode_categorical(dataset, categorical_columns)

        y_pred = self.bestModel.predict(dataset)
        return y_pred


if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    # List of CSV files
    file_paths = [
    '/home/iman/projects/kara/Projects/T-Rize/archive/City_data/subset_0.csv',
    '/home/iman/projects/kara/Projects/T-Rize/archive/City_data/subset_1.csv',
    '/home/iman/projects/kara/Projects/T-Rize/archive/City_data/subset_2.csv',
    '/home/iman/projects/kara/Projects/T-Rize/archive/City_data/subset_3.csv'
]

    # Loading data and combining into one dataframe
    dfs = [pd.read_csv(file) for file in file_paths]
    dataset = pd.concat(dfs, ignore_index=True)

    # Extracting features and target
    dataset = dataset.drop(columns=['Address', 'City', 'State', 'County', 'Zip Code', 'Latitude', 'Longitude'])
    X = dataset.drop(columns=['Price'])  # Features
    y = dataset['Price'] 

    # Splitting data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Creating and using PSO_XGBoost_regressor model
    popSize = 25
    maxIt = 200
    regressor = PSO_XGBoost_regressor(x_train, y_train, popSize, maxIt)
    regressor.fit(isPlot=True)

    # Predicting on test data
    y_pred = regressor.predict(x_test)

    # Calculating evaluation metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print('best cost: ', regressor.gbestCost)
    print('best solution: ', regressor.gbestPos)
    print('r2 score: ', r2, ' mean squared error: ', mse)
