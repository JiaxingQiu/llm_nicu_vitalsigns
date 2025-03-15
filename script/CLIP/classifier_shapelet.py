import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


class ShapeletClassifier:
    def __init__(self, min_length=10, max_length=50, n_shapelets=5, step_size=5, random_state=42):
        self.min_length = min_length
        self.max_length = max_length
        self.n_shapelets = n_shapelets
        self.step_size = step_size
        self.random_state = random_state
        self.shapelets = []
        self.classifier = LogisticRegression(random_state=random_state)
        
    def _extract_candidates(self, ts_data):
        candidates = []
        for ts in ts_data:
            for length in range(self.min_length, self.max_length, self.step_size):
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
    def __init__(self, min_length=10, max_length=50, n_shapelets=5, step_size=5,
                 random_state=42, n_jobs=-1):
        super().__init__(min_length, max_length, n_shapelets, step_size, random_state)
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



# from joblib import Parallel, delayed
# from tqdm import tqdm

# class FastShapeletClassifier(ShapeletClassifier):
#     def __init__(self, min_length=10, max_length=50, n_shapelets=5, 
#                  random_state=42, n_jobs=-1):
#         super().__init__(min_length, max_length, n_shapelets, random_state)
#         self.n_jobs = n_jobs
    
#     def _evaluate_candidate(self, candidate_tuple, X, y):
#         """Evaluate a single shapelet candidate"""
#         shapelet, length, start = candidate_tuple
#         # Calculate distances to this shapelet
#         distances = np.array([self._calculate_distance(shapelet, ts) for ts in X])
#         # Calculate information gain
#         gain, threshold = self._calculate_information_gain(distances, y)
#         return (shapelet, gain, threshold)
    
#     def fit(self, X, y):
#         """
#         Parallel version of shapelet fitting using joblib
#         """
#         print("Extracting candidates...")
#         candidates = self._extract_candidates(X)
        
#         print(f"Evaluating {len(candidates)} shapelets in parallel...")
#         shapelet_scores = Parallel(n_jobs=self.n_jobs, verbose=1)(
#             delayed(self._evaluate_candidate)(candidate, X, y) 
#             for candidate in candidates
#         )
        
#         print("Selecting top shapelets...")
#         shapelet_scores.sort(key=lambda x: x[1], reverse=True)
#         self.shapelets = shapelet_scores[:self.n_shapelets]
        
#         print("Training classifier...")
#         X_transformed = self._transform(X)
#         self.classifier.fit(X_transformed, y)
#         return self

# # Usage:
# """
# clf = FastShapeletClassifier(
#     min_length=30,    # 30 seconds
#     max_length=300,   # 5 minutes
#     n_shapelets=5,
#     n_jobs=-1  # Use all available cores
# )
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# """




def plot_shapelet_matches(shapelet, X_train, y_train=None,n_matches=5, window_size=20):
    """
    Plot time series that best match a shapelet and highlight the matching regions.
    
    Parameters:
    -----------
    shapelet : array-like
        The shapelet pattern to match
    X_train : array-like
        Training time series data
    n_matches : int
        Number of best matches to show
    window_size : int
        Additional context window size around the match
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    def find_best_match_position(shapelet, ts):
        """Find the position where the shapelet best matches in the time series"""
        min_dist = float('inf')
        best_pos = 0
        shapelet_length = len(shapelet)
        
        for i in range(len(ts) - shapelet_length + 1):
            dist = euclidean(shapelet, ts[i:i + shapelet_length])
            if dist < min_dist:
                min_dist = dist
                best_pos = i
        return best_pos, min_dist
    
    # Calculate distances and positions for all time series
    distances = []
    positions = []
    for i, ts in enumerate(X_train):
        pos, dist = find_best_match_position(shapelet, ts)
        distances.append(dist)
        positions.append((i, pos, dist))
    
    # Sort by distance and get top n_matches
    best_matches = sorted(positions, key=lambda x: x[2])[:n_matches]
    
    # Plot
    fig, axes = plt.subplots(n_matches + 1, 1, figsize=(15, 3*(n_matches + 1)))
    
    # Plot shapelet itself
    axes[0].plot(shapelet, 'r-', linewidth=2)
    axes[0].set_title('Shapelet Pattern')
    axes[0].grid(True)
    
    # Plot each matching time series
    for i, (ts_idx, pos, dist) in enumerate(best_matches, 1):
        ts = X_train[ts_idx]
        
        # Determine plot range with context window
        start_idx = max(0, pos - window_size)
        end_idx = min(len(ts), pos + len(shapelet) + window_size)
        
        # Plot full segment
        axes[i].plot(range(start_idx, end_idx), 
                    ts[start_idx:end_idx], 
                    'b-', 
                    label='Time Series')
        
        # Highlight matching segment
        axes[i].plot(range(pos, pos + len(shapelet)), 
                    ts[pos:pos + len(shapelet)], 
                    'r-', 
                    linewidth=2, 
                    label='Match')
        if y_train is not None:
            axes[i].set_title(f'Match {i}: Time Series {ts_idx}, Distance: {dist:.3f}, Class: {y_train[ts_idx]}')
        else:
            axes[i].set_title(f'Match {i}: Time Series {ts_idx}, Distance: {dist:.3f}')
        axes[i].grid(True)
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()
    
    return best_matches

# # Example usage:
# # Get the first shapelet
# shapelet1 = clf.get_shapelets()[0][0]  # Get first shapelet pattern
# matches = plot_shapelet_matches(shapelet1, X_train, n_matches=5)

# # Print detailed information about matches
# print("\nBest Matching Time Series:")
# for i, (ts_idx, pos, dist) in enumerate(matches, 1):
#     print(f"Match {i}:")
#     print(f"- Time Series Index: {ts_idx}")
#     print(f"- Starting Position: {pos}")
#     print(f"- Distance: {dist:.3f}")



def plot_all_shapelet_matches_grid(clf, X_train, y_train, n_matches=5, window_size=None):
    """
    Plot all shapelets and their matches in a grid layout.
    
    Parameters:
    -----------
    clf : ShapeletClassifier
        Trained shapelet classifier
    X_train : array-like
        Training time series data
    y_train : array-like
        Training labels
    n_matches : int
        Number of best matches to show per shapelet
    window_size : int
        Additional context window size around the match
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    def find_best_match_position(shapelet, ts):
        """Find the position where the shapelet best matches in the time series"""
        min_dist = float('inf')
        best_pos = 0
        shapelet_length = len(shapelet)
        
        for i in range(len(ts) - shapelet_length + 1):
            dist = euclidean(shapelet, ts[i:i + shapelet_length])
            if dist < min_dist:
                min_dist = dist
                best_pos = i
        return best_pos, min_dist
    
    # Get all shapelets
    shapelets = clf.get_shapelets()
    n_shapelets = len(shapelets)
    
    # Create figure with grid layout
    fig, axes = plt.subplots(n_matches + 1, n_shapelets, 
                            figsize=(6*n_shapelets, 3*(n_matches + 1)))
    
    # Process each shapelet
    for shapelet_idx, (shapelet, score) in enumerate(shapelets):
        # Calculate distances and positions for all time series
        positions = []
        for i, ts in enumerate(X_train):
            pos, dist = find_best_match_position(shapelet, ts)
            positions.append((i, pos, dist))
        
        # Sort by distance and get top n_matches
        best_matches = sorted(positions, key=lambda x: x[2])[:n_matches]
        
        # Plot shapelet itself in first row
        axes[0, shapelet_idx].plot(shapelet, 'r-', linewidth=2)
        axes[0, shapelet_idx].set_title(f'Shapelet {shapelet_idx+1}\nScore: {score:.3f}')
        axes[0, shapelet_idx].grid(True)
        
        # Plot each matching time series
        for match_idx, (ts_idx, pos, dist) in enumerate(best_matches, 1):
            ts = X_train[ts_idx]
            
            if window_size is None:
                # Plot entire time series
                start_idx = 0
                end_idx = len(ts)
            else:
                # Plot windowed segment
                start_idx = max(0, pos - window_size)
                end_idx = min(len(ts), pos + len(shapelet) + window_size)
            
            # Plot full segment
            axes[match_idx, shapelet_idx].plot(
                range(start_idx, end_idx), 
                ts[start_idx:end_idx], 
                'b-', 
                label='Time Series'
            )
            
            # Highlight matching segment
            axes[match_idx, shapelet_idx].plot(
                range(pos, pos + len(shapelet)), 
                ts[pos:pos + len(shapelet)], 
                'r-', 
                linewidth=2, 
                label='Match'
            )
            
            axes[match_idx, shapelet_idx].set_title(
                f'Match {match_idx}: TS {ts_idx}\nDist: {dist:.3f}, Class: {y_train[ts_idx]}'
            )
            axes[match_idx, shapelet_idx].grid(True)
            
            # Only show legend for first column
            if shapelet_idx == 0:
                axes[match_idx, shapelet_idx].legend()
    
    plt.tight_layout()
    plt.show()
    
    # # Print summary information
    # print("\nShapelet Match Summary:")
    # for shapelet_idx, (shapelet, score) in enumerate(shapelets):
    #     print(f"\nShapelet {shapelet_idx + 1}:")
    #     print(f"Score: {score:.3f}")
    #     print(f"Length: {len(shapelet)}")
        
    #     # Calculate class distribution of best matches
    #     match_classes = [y_train[pos[0]] for pos in sorted(positions, key=lambda x: x[2])[:n_matches]]
    #     class_counts = np.bincount(match_classes)
    #     print("Class distribution of best matches:")
    #     for class_idx, count in enumerate(class_counts):
    #         print(f"Class {class_idx}: {count}")

# Example usage:
# plot_all_shapelet_matches_grid(clf, X_train, y_train, n_matches=5, window_size=20)


def analyze_shapelet_discrimination(clf, X_train, y_train):
    """
    Analyze how well each shapelet discriminates between classes.
    
    Parameters:
    -----------
    clf : ShapeletClassifier
        Trained shapelet classifier
    X_train : array-like
        Training time series data
    y_train : array-like
        Training labels
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    def calculate_distances(shapelet, X):
        """Calculate minimum distances between shapelet and all time series"""
        distances = []
        for ts in X:
            min_dist = float('inf')
            for i in range(len(ts) - len(shapelet) + 1):
                dist = euclidean(shapelet, ts[i:i + len(shapelet)])
                min_dist = min(min_dist, dist)
            distances.append(min_dist)
        return np.array(distances)
    
    # Get all shapelets
    shapelets = clf.get_shapelets()
    n_shapelets = len(shapelets)
    classes = np.unique(y_train)
    
    # Create figure for boxplots
    fig, axes = plt.subplots(1, n_shapelets, figsize=(6*n_shapelets, 5))
    if n_shapelets == 1:
        axes = [axes]
    
    # Store discrimination metrics
    shapelet_metrics = []
    
    # Analyze each shapelet
    for idx, (shapelet, score) in enumerate(shapelets):
        # Calculate distances for all time series
        distances = calculate_distances(shapelet, X_train)
        
        # Calculate distance distributions per class
        class_distances = {c: distances[y_train == c] for c in classes}
        
        # Calculate statistics
        class_stats = {}
        for c in classes:
            dist = class_distances[c]
            class_stats[c] = {
                'mean': np.mean(dist),
                'std': np.std(dist)
            }
        
        # Perform t-test between classes
        t_stat, p_value = stats.ttest_ind(
            class_distances[0],
            class_distances[1]
        )
        
        # Plot boxplot
        axes[idx].boxplot([class_distances[c] for c in classes])
        axes[idx].set_xticklabels(['Survival', 'Death'])
        axes[idx].set_title(f'Shapelet {idx+1}\nScore: {score:.3f}\np-value: {p_value:.3e}')
        axes[idx].set_ylabel('Distance')
        axes[idx].grid(True)
        
        # Calculate discrimination metrics
        preferred_class = 0 if class_stats[0]['mean'] < class_stats[1]['mean'] else 1
        discrimination_strength = abs(class_stats[0]['mean'] - class_stats[1]['mean']) / \
                                (class_stats[0]['std'] + class_stats[1]['std'])
        
        shapelet_metrics.append({
            'shapelet_idx': idx,
            'preferred_class': preferred_class,
            'discrimination_strength': discrimination_strength,
            'p_value': p_value,
            'class_0_mean': class_stats[0]['mean'],
            'class_1_mean': class_stats[1]['mean'],
            'score': score
        })
    
    plt.tight_layout()
    plt.show()
    
    # # Print detailed analysis
    # print("\nShapelet Discrimination Analysis:")
    # print("="*50)
    # for metric in shapelet_metrics:
    #     print(f"\nShapelet {metric['shapelet_idx'] + 1}:")
    #     print(f"Indicator pattern for: {'Death' if metric['preferred_class'] == 1 else 'Survival'}")
    #     print(f"Discrimination strength: {metric['discrimination_strength']:.3f}")
    #     print(f"Statistical significance: p = {metric['p_value']:.3e}")
    #     print(f"Mean distance for Death: {metric['class_0_mean']:.3f}")
    #     print(f"Mean distance for Survival: {metric['class_1_mean']:.3f}")
    #     print(f"Information gain score: {metric['score']:.3f}")
    
    return shapelet_metrics

# # Example usage:
# metrics = analyze_shapelet_discrimination(clf, X_train, y_train)