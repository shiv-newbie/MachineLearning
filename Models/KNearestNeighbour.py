import statistics

class KNearestNeighbour:
    def __init__(self):
        pass
    
    def train(self, X, y):
        self.X = X.to_numpy()
        self.y = y.to_numpy()
        
    def predict(self, X_test, k):
        # Number of columns in the test
        num_test = X_test.shape[0]
        
        y_pred = np.zeros(num_test, dtype=self.y.dtype)
        
        for i in range(num_test):
            
            num_test = y_test.to_numpy()
            
            # Get the : L1 distance (Manhattan Distance)
            distances = np.sum(np.abs(self.X-num_test[i]), axis = 1)
            
            # Get the : L2 distance (Euclidian Distance)
            # distances = np.sqrt(np.sum((self.X-num_test[i])**2, axis = 1))
            
            # Get the index of the min value from all the distances
            # Find the indices of the n nearest neighbors
            min_indices = np.argsort(distances)[:k]
            kNearest = self.y[min_indices]
            try:
                mode_value = statistics.mode(kNearest)
            except statistics.StatisticsError:
                mode_value = self.y[min_indices[0]]

            y_pred[i] = mode_value
            
        return pd.Series(y_pred)
        
    def accuracy(self, y_test, y_pred):
        # Reset the indices of y_test and y_pred
        y_test_reset = y_test.reset_index(drop=True)
        y_pred_reset = pd.Series(y_pred)

        # Convert both to NumPy arrays
        y_test_array = y_test_reset.to_numpy()
        y_pred_array = y_pred_reset.to_numpy()

        # Now, you can compare y_test_array and y_pred_array
        correct = (y_test_array == y_pred_array).sum()
        accuracy = float(correct)*100 / float(len(y_test_reset))

        print(f'KNearestNeighbour accuracy: {accuracy:.2f}%')
