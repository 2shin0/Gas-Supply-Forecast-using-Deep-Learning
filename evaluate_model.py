import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def evaluate_model(model, X_test, y_test):
    y_test_pred = model.predict(X_test)

    test_loss = mean_squared_error(y_test, y_test_pred)

    print("Test Loss:", test_loss)

    return y_test, y_test_pred

def plot_learning_curve(history):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_predictions(y_test, y_test_pred):
    plt.figure()
    plt.plot(y_test, color='red', label='real target y')
    plt.plot(y_test_pred, color='blue', label='predict y')
    plt.legend()
    plt.show()