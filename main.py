from preprocess_data import preprocess_total_data
from train_model import train_lstm_model, create_sequences
from evaluate_model import evaluate_model, plot_learning_curve, plot_predictions
from sklearn.preprocessing import MinMaxScaler

def main():
    total_path = "./dataset/한국가스공사_시간별 공급량_20181231.csv"
    temp_path = "./dataset/avg_temp.csv"

    total = preprocess_total_data(total_path, temp_path)

    data = total[['month', 'temp', 'supply']]

    scaler = MinMaxScaler()
    scaler.fit(data)
    model, history = train_lstm_model(data)

    plot_learning_curve(history)

    window_size = 30
    X_test, y_test = create_sequences(scaler.transform(data), window_size)
    y_test, y_test_pred = evaluate_model(model, X_test, y_test)

    plot_predictions(y_test, y_test_pred)

if __name__ == "__main__":
    main()