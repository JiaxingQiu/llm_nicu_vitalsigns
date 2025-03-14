import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


class ShapeletClassifier:
    def __init__(self, min_length=10, max_length=50, n_shapelets=5, random_state=42):
        self.min_length = min_length
        self.max_length = max_length
        self.n_shapelets = n_shapelets
        self.random_state = random_state
        self.shapelets = []
        self.classifier = LogisticRegression(random_state=random_state)
        
    def _extract_candidates(self, ts_data, step_size=5):
        candidates = []
        for ts in ts_data:
            for length in range(self.min_length, self.max_length, step_size):
                for start in range(len(ts) - length + 1):
                    shapelet = ts[start:start + length]
                    candidates.append((shapelet, length, start))
        return candidates
    
    def _calculate_distance(self, shapelet, ts):
        min_dist = float('inf')
        shapelet_length = len(shapelet)
        
        for i in range(len(ts) - shapelet_length + 1):
            dist = euclidean(shapelet, ts[i:i + shapelet_length])
            min_dist = min(min_dist, dist)
        return min_dist
    
    def _calculate_information_gain(self, distances, labels):
        # Sort distances and corresponding labels
        sorted_idx = np.argsort(distances)
        sorted_distances = distances[sorted_idx]
        sorted_labels = labels[sorted_idx]
        
        # Calculate information gain at each possible split point
        total_entropy = self._entropy(labels)
        best_gain = -float('inf')
        best_threshold = None
        
        for i in range(1, len(distances)):
            threshold = (sorted_distances[i-1] + sorted_distances[i]) / 2
            left_labels = sorted_labels[:i]
            right_labels = sorted_labels[i:]
            
            left_entropy = self._entropy(left_labels)
            right_entropy = self._entropy(right_labels)
            
            # Calculate weighted average entropy
            n = len(labels)
            weighted_entropy = (len(left_labels)/n * left_entropy + 
                              len(right_labels)/n * right_entropy)
            gain = total_entropy - weighted_entropy
            
            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold
                
        return best_gain, best_threshold
    
    def _entropy(self, labels):
        if len(labels) == 0:
            return 0
        p1 = np.mean(labels)
        if p1 == 0 or p1 == 1:
            return 0
        p0 = 1 - p1
        return -p0 * np.log2(p0) - p1 * np.log2(p1)
    
    def fit(self, X, y):
        """
        X: array-like of shape (n_samples, n_timesteps)
        y: array-like of shape (n_samples,)
        """
        # Extract shapelet candidates
        candidates = self._extract_candidates(X)
        
        # Evaluate each candidate
        shapelet_scores = []
        for shapelet, length, start in candidates:
            # Calculate distances to this shapelet
            distances = np.array([self._calculate_distance(shapelet, ts) for ts in X])
            
            # Calculate information gain
            gain, threshold = self._calculate_information_gain(distances, y)
            shapelet_scores.append((shapelet, gain, threshold))
        
        # Select top n_shapelets
        shapelet_scores.sort(key=lambda x: x[1], reverse=True)
        self.shapelets = shapelet_scores[:self.n_shapelets]
        
        # Transform data using selected shapelets
        X_transformed = self._transform(X)
        
        # Train classifier
        self.classifier.fit(X_transformed, y)
        return self
    
    def _transform(self, X):
        """Transform time series using shapelet distances"""
        X_transformed = np.zeros((len(X), len(self.shapelets)))
        for i, ts in enumerate(X):
            for j, (shapelet, _, _) in enumerate(self.shapelets):
                X_transformed[i, j] = self._calculate_distance(shapelet, ts)
        return X_transformed
    
    def predict(self, X):
        """Predict using the shapelet classifier"""
        X_transformed = self._transform(X)
        return self.classifier.predict(X_transformed)
    
    def predict_proba(self, X):
        """Predict probability estimates"""
        X_transformed = self._transform(X)
        return self.classifier.predict_proba(X_transformed)
    
    def get_shapelets(self):
        """Return the learned shapelets and their information gain scores"""
        return [(shapelet, score) for shapelet, score, _ in self.shapelets]

# Usage example:
"""
# Initialize classifier
clf = ShapeletClassifier(
    min_length=20,
    max_length=50,
    n_shapelets=5
)

# Fit classifier
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Get performance metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Get learned shapelets
shapelets = clf.get_shapelets()

# Visualize shapelets
import matplotlib.pyplot as plt

def plot_shapelets(shapelets, title="Discovered Shapelets"):
    fig, axes = plt.subplots(len(shapelets), 1, figsize=(10, 2*len(shapelets)))
    for i, (shapelet, score) in enumerate(shapelets):
        axes[i].plot(shapelet)
        axes[i].set_title(f"Shapelet {i+1}, Information Gain: {score:.4f}")
    plt.tight_layout()
    plt.show()

plot_shapelets(shapelets)
""" 


from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

class FastShapeletClassifier(ShapeletClassifier):
    def __init__(self, min_length=10, max_length=50, n_shapelets=5, 
                 random_state=42, n_jobs=-1):
        super().__init__(min_length, max_length, n_shapelets, random_state)
        self.n_jobs = n_jobs
    
    def _evaluate_candidate(self, candidate_tuple):
        """Evaluate a single shapelet candidate"""
        shapelet, length, start = candidate_tuple
        # Calculate distances to this shapelet
        distances = np.array([self._calculate_distance(shapelet, ts) for ts in self.X])
        # Calculate information gain
        gain, threshold = self._calculate_information_gain(distances, self.y)
        return (shapelet, gain, threshold)
    
    def fit(self, X, y):
        """
        Parallel version of shapelet fitting
        """
        print("Extracting candidates...")
        candidates = self._extract_candidates(X)
        
        # Store X and y as instance variables for parallel processing
        self.X = X
        self.y = y
        
        print(f"Evaluating {len(candidates)} shapelets in parallel...")
        with Pool(processes=self.n_jobs) as pool:
            shapelet_scores = list(tqdm(
                pool.imap(self._evaluate_candidate, candidates),
                total=len(candidates),
                desc="Finding shapelets"
            ))
        
        # Clean up instance variables
        del self.X
        del self.y
        
        print("Selecting top shapelets...")
        shapelet_scores.sort(key=lambda x: x[1], reverse=True)
        self.shapelets = shapelet_scores[:self.n_shapelets]
        
        print("Training classifier...")
        X_transformed = self._transform(X)
        self.classifier.fit(X_transformed, y)
        return self

# Usage:
"""
clf = FastShapeletClassifier(
    min_length=30,    # 30 seconds
    max_length=300,   # 5 minutes
    n_shapelets=5,
    n_jobs=-1  # Use all available cores
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
"""