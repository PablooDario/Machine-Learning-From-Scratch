import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

class LogisticRegressionVisualizer:
    def __init__(self):
        # Define colors
        self.color_class0 = '#ffffcc' 
        self.color_class1 = '#008B8B' 
        self.cmap_prob = LinearSegmentedColormap.from_list('custom_cmap', 
                                                          [self.color_class1, self.color_class0])
        
    def generate_1d_data(self, seed=0):
        """Generate 1D classification data"""
        np.random.seed(seed)
        x_class0 = np.random.normal(2, 0.2, 50)
        x_class1 = np.random.normal(5, 0.2, 50)
        X_train = np.concatenate([x_class0, x_class1]).reshape(-1, 1)
        y_train = np.array([1] * 50 + [0] * 50)
        return x_class0, x_class1, X_train, y_train
    
    def generate_2d_data(self, n_points=100, seed=0):
        """Generate 2D classification data"""
        np.random.seed(seed)
        x1_class0 = np.random.normal(2, 1, n_points)
        x2_class0 = np.random.normal(2, 1, n_points)
        x1_class1 = np.random.normal(5, 1, n_points)
        x2_class1 = np.random.normal(5, 1, n_points)
        
        X = np.vstack((np.column_stack((x1_class0, x2_class0)), np.column_stack((x1_class1, x2_class1))))
        y = np.hstack((np.zeros(n_points), np.ones(n_points)))
        return X, y
    
    def plot_1d_training_data(self, ax, x_class0, x_class1):
        """Plot 1D training data"""
        ax.scatter(x_class0, [1] * len(x_class0), color=self.color_class0, label='Class 0', s=60, edgecolor='k')
        ax.scatter(x_class1, [0] * len(x_class1), color=self.color_class1, label='Class 1', s=60, edgecolor='k')
        ax.set_xlabel('Feature', fontsize=14)
        ax.set_ylabel('Class', fontsize=14)
        ax.set_title('Training Data', fontsize=16)
        ax.legend(fontsize=12)
        
    def plot_decision_boundary(self, ax, model, X_train, x_class0, x_class1):
        """Plot decision boundary and sigmoid curve"""
        x_min, x_max = X_train.min() - 0.5, X_train.max() + 0.5
        y_min, y_max = -3, 3.1
        
        # Create test data and mesh grid
        X_test = np.linspace(1, 6, 100).reshape(-1, 1)
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
        
        # Get predictions
        y_test_probs = model.predict_proba(X_test)[:, 1]
        Z = model.predict_proba(xx.reshape(-1, 1))[:, 1].reshape(xx.shape)
        decision_boundary = (-model.intercept_ / model.coef_).item()
        
        # Plot
        ax.scatter(x_class0, [1] * len(x_class0), color=self.color_class0, s=80, edgecolor='k', linewidth=1.5)
        ax.scatter(x_class1, [0] * len(x_class1), color=self.color_class1, s=80, edgecolor='k', linewidth=1.5)
        ax.contourf(xx, yy, Z, levels=50, cmap=self.cmap_prob, alpha=0.3)
        ax.axvline(x=decision_boundary, color='black', linestyle='--', label='Decision Boundary')
        ax.plot(X_test, y_test_probs, color='darkorange', linewidth=2.5, label='Sigmoid Function')
        
        ax.set_xlabel('Feature', fontsize=14)
        ax.set_ylabel('Probability of Class 1', fontsize=14)
        ax.set_title('Classification Region and Decision Boundary', fontsize=16)
        ax.legend(fontsize=12)
        ax.set_xlim(x_min + 0.3, x_max - 0.3)
        ax.set_ylim(-0.1, 1.1)
        
    def plot_decision_mapping(self, model, X_train):
        """Plot decision mapping visualization"""
        plt.figure(figsize=(10, 6))
        
        x_min, x_max = X_train.min() - 0.5, X_train.max() + 0.5
        y_min, y_max = -3, 3.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), 
                            np.linspace(y_min, y_max, 10))
        Z = model.predict_proba(xx.reshape(-1, 1))[:, 1].reshape(xx.shape)
        
        # Plot decision boundary and sigmoid
        X_test = np.linspace(1, 6, 100).reshape(-1, 1)
        y_test_probs = model.predict_proba(X_test)[:, 1]
        plt.plot(X_train, (X_train * model.coef_ + model.intercept_),  color='black', label='Linear Combination')
        plt.plot(X_test, y_test_probs, color='darkorange', linewidth=2.5, label='Sigmoid Curve')
        plt.contourf(xx, yy, Z, levels=50, cmap=self.cmap_prob, alpha=0.3)
        
        # Plot test points
        X_test = np.arange(2.6, 4.9, 0.1).reshape(-1, 1)
        probs = model.predict_proba(X_test)[:, 1]
        plt.scatter(X_test[probs > 0.5], probs[probs > 0.5], c='white', edgecolor='k', s=80, label='Class 0')
        plt.scatter(X_test[probs <= 0.5], probs[probs <= 0.5], c='black', edgecolor='k', s=80, label='Class 1')
        
        # Plot decision boundary points
        decision_values = (X_test * model.coef_ + model.intercept_).flatten()
        plt.scatter(X_test, decision_values, c=probs, cmap=self.cmap_prob, edgecolor='k', s=80)
        
        # Plot distance lines
        for i in range(len(X_test)):
            plt.plot([X_test[i], X_test[i]], [decision_values[i], probs[i]], 
                    'k--', lw=0.8)
        
        plt.xlim(2.5, 4.65)
        plt.ylim(y_min, y_max)
        plt.xlabel('$x1$ (Feature)')
        plt.ylabel('Linear Map to Probability')
        plt.title('Distance between the linear combination and the sigmoid')
        plt.legend()
        plt.grid(True)
        plt.colorbar(label='Probability')
        plt.show()
        
    def create_animation(self, model, X_train, y_train):
        """Create animation of sigmoid curve fitting"""
        X_test = np.arange(2.6, 4.8, 0.2).reshape(-1, 1)
        
        # Collect training history
        probs_history = []
        distances_history = []
        for i in range(1, model.n_iter_[0] + 1):
            temp_model = LogisticRegression(warm_start=True, max_iter=i)
            temp_model.fit(X_train, y_train)
            probs_history.append(temp_model.predict_proba(X_test)[:, 1])
            distances_history.append(np.abs(probs_history[-1] - y_train[:len(X_test)]))
        
        # Set up animation
        fig, ax = plt.subplots(figsize=(10, 6))
        sigmoid_line, = ax.plot([], [], lw=2)
        distance_lines = []
        
        # Initial plot setup
        ax.scatter([2.4, 2.6, 2.8, 3, 3.2, 3.4], [1] * 6, color=self.color_class1, 
                  s=80, edgecolor="k", linewidth=1.5)
        ax.scatter([3.6, 3.8, 4, 4.2, 4.4], [0] * 5, color=self.color_class0, 
                  s=80, edgecolor="k", linewidth=1.5)
        
        # Plot configuration
        ax.set_xlim(2.5, 4.45)
        ax.set_ylim(-0.2, 1.2)
        ax.set_xlabel("Feature (x1)")
        ax.set_ylabel("Probability")
        ax.set_title("Sigmoid Curve Fitting During Training")
        ax.grid(True)
        
        # Add background
        xx, yy = np.meshgrid(np.linspace(2.5, 4.5, 100), np.linspace(-0.2, 1.2, 100))
        Z = model.predict_proba(np.c_[xx.ravel()])[:, 1].reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=50, cmap="Blues", alpha=0.3)
        
        def init():
            sigmoid_line.set_data([], [])
            for line in distance_lines:
                line.remove()
            distance_lines.clear()
            return [sigmoid_line] + distance_lines
        
        def update(epoch):
            prob = probs_history[epoch]
            distances = distances_history[epoch]
            sigmoid_line.set_data(X_test, prob)
            
            for line in distance_lines:
                line.remove()
            distance_lines.clear()
            
            for x, p, dist in zip(X_test.flatten(), prob, distances):
                line = ax.plot([x, x], 
                             [p, y_train[np.argmin(np.abs(X_train.flatten() - x))]], 
                             color="gray", linestyle="--")
                distance_lines.append(line[0])
            
            return [sigmoid_line] + distance_lines
        
        ani = FuncAnimation(fig, update, frames=len(probs_history), 
                          init_func=init, blit=True, interval=500)
        
        # Save the dynamic visualization as gif
        ani.save("SupervisedLearning/02 Logistic Regression/imgs/SigmoidCurveFitting.gif", writer="pillow", fps=7)
        
        plt.show()
        
    def plot_2d_classification(self, X, y, model):
        """Plot 2D classification visualization"""
        # Create grid for decision boundary
        x1_range = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
        x2_range = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
        xx1, xx2 = np.meshgrid(x1_range, x2_range)
        grid = np.c_[xx1.ravel(), xx2.ravel()]
        probs = model.predict_proba(grid)[:, 1].reshape(xx1.shape)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 2D Plot
        ax1.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color=self.color_class1, label='Class 0', s=80, edgecolor='k', linewidth=1.5)
        ax1.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color=self.color_class0, label='Class 1', s=80, edgecolor='k', linewidth=1.5)
        ax1.contour(xx1, xx2, probs, levels=[0.5], linewidths=2, colors='black')
        ax1.set_xlabel('$x_1$', fontsize=14)
        ax1.set_ylabel('$x_2$', fontsize=14)
        ax1.legend(fontsize=12)
        ax1.set_title('Decision Boundary in 2D', fontsize=16)
        
        # 3D Plot
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(X[y == 0][:, 0], X[y == 0][:, 1], y[y == 0], 
                   color=self.color_class1, label='Class 0', s=80, 
                   edgecolor='k', linewidth=1.5)
        ax2.scatter(X[y == 1][:, 0], X[y == 1][:, 1], y[y == 1], 
                   color=self.color_class0, label='Class 1', s=80, 
                   edgecolor='k', linewidth=1.5)
        ax2.plot_surface(xx1, xx2, probs, cmap=self.cmap_prob, 
                        alpha=0.6, edgecolor='none')
        
        ax2.set_xlabel('$x_1$', fontsize=14)
        ax2.set_ylabel('$x_2$', fontsize=14)
        ax2.set_zlabel('Probability', fontsize=14)
        ax2.set_title('Sigmoid in 3D', fontsize=16)
        ax2.view_init(elev=10, azim=155)
        
        plt.tight_layout()
        plt.show()

def main():
    # Initialize visualizer
    viz = LogisticRegressionVisualizer()
    
    # Generate and fit 1D data
    x_class0, x_class1, X_train, y_train = viz.generate_1d_data()
    model_1d = LogisticRegression()
    model_1d.fit(X_train, y_train)
    
    # Plot 1D visualizations
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    viz.plot_1d_training_data(axs[0], x_class0, x_class1)
    viz.plot_decision_boundary(axs[1], model_1d, X_train, x_class0, x_class1)
    plt.tight_layout()
    plt.show()
    
    # Plot decision mapping
    viz.plot_decision_mapping(model_1d, X_train)
    
    # Create animation
    viz.create_animation(model_1d, X_train, y_train)
    
    # Generate and fit 2D data
    X_2d, y_2d = viz.generate_2d_data()
    model_2d = LogisticRegression()
    model_2d.fit(X_2d, y_2d)
    
    # Plot 2D visualization
    viz.plot_2d_classification(X_2d, y_2d, model_2d)

if __name__ == "__main__":
    main()