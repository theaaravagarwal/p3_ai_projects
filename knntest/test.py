from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#random data gen
np.random.seed(67)
class0_x = np.random.uniform(0, 5, size=50); class0_y = np.random.uniform(0, 5, size=50)
class1_x = np.random.uniform(5, 10, size=50); class1_y = np.random.uniform(5, 10, size=50)
class0_label = [0]*50; class1_label = [1]*50
data = {'x': np.concatenate([class0_x, class1_x]),'y': np.concatenate([class0_y, class1_y]),'label': class0_label+class1_label}
df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=67).reset_index(drop=True)

#features and labels
X = df[['x', 'y']]
y = df['label']

#knn itself
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X, y)

#new points
new_points = pd.DataFrame({'x': [4, 5], 'y': [4, 5]})
predictions = knn.predict(new_points)

print("Predictions for new points:")
print(predictions)

plt.figure(figsize=(8, 6))

x_min, x_max = df['x'].min()-1, df['x'].max()+1 #create a mesh grid chat
y_min, y_max = df['y'].min()-1, df['y'].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

#predict on mesh grid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

#plot decision bounds
plt.contourf(xx, yy, Z, alpha=0.2, cmap='bwr')

#plot og pts
plt.scatter(df['x'], df['y'], c=df['label'], cmap='bwr', edgecolor='k', s=100, label='Training points')

#plot new pts
plt.scatter(new_points['x'], new_points['y'], c=predictions, cmap='cool', edgecolor='k', marker='X', s=200, label='New predictions')

#annot new pts
for i, point in new_points.iterrows():
    plt.text(point['x'] + 0.1, point['y'] + 0.1, f"Class {predictions[i]}", fontsize=12, fontweight='bold')

plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'KNN Classification Demo (k={k})')
plt.legend()
plt.grid(True)
plt.show()