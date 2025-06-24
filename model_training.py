from utils.data_loader import SequentialImageDataGenerator
from utils.model import NextImagePredictor
from utils.visualization import plot_loss

# Data
generator = SequentialImageDataGenerator()
X_train, X_test, y_train, y_test = generator.re_scale_data()

# Model
model = NextImagePredictor(X_train.shape[1:])
history = model.train_model(X_train, y_train, optimizer='adam', epochs=10)

# Save model
model.save("final_model_adam.h5")

# Visualisasi
plot_loss(history, 'adam')

# Evaluasi
loss, mse = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MSE: {mse:.6f}")