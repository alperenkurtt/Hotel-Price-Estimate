from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os

from train_model import train_and_save_models

app = Flask(__name__)

# Global değişkenler
model = None
model_info = None
unique_values = None


def load_model_and_data():
    global model, model_info, unique_values

    # Eğer model yoksa, eğit ve kaydet
    if not os.path.exists('models/model_info.pkl'):
        model_info = train_and_save_models()
    else:
        # Model bilgilerini yükle
        with open('models/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)

    # Modeli yükle
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Benzersiz değerleri yükle
    with open('models/unique_values.pkl', 'rb') as f:
        unique_values = pickle.load(f)

    print(f"Model yüklendi: {model_info['model_name']}")
    print(f"Model R² skoru: {model_info['r2']:.4f}")


# Uygulama başladığında modeli yükle
load_model_and_data()


# Ana sayfa
@app.route('/')
def index():
    return render_template('index.html', model_info=model_info)


# Tahmin sayfası
@app.route('/predict')
def predict_page():
    return render_template('predict.html',
                           meal_plans=unique_values['meal_plans'],
                           room_types=unique_values['room_types'],
                           market_segments=unique_values['market_segments'])


# Tahmin API'si
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Form verilerini al
        data = request.form.to_dict()
        print(f"Alınan veriler: {data}")

        # Veriyi modelin beklediği formata dönüştür
        input_data = {}

        # Sayısal değerleri dönüştür
        numerical_fields = {
            'no_of_adults': int,
            'no_of_children': int,
            'no_of_weekend_nights': int,
            'no_of_week_nights': int,
            'required_car_parking_space': int,
            'lead_time': int,
            'arrival_year': int,
            'arrival_month': int,
            'arrival_date': int,
            'repeated_guest': int,
            'no_of_previous_cancellations': int,
            'no_of_previous_bookings_not_canceled': int,
            'no_of_special_requests': int,
            'booking_status': int
        }

        for field, dtype in numerical_fields.items():
            try:
                input_data[field] = dtype(data.get(field, 0))
            except (ValueError, TypeError):
                input_data[field] = 0

        # Kategorik değerleri ekle
        input_data['type_of_meal_plan'] = data.get('type_of_meal_plan', 'Meal Plan 1')
        input_data['room_type_reserved'] = data.get('room_type_reserved', 'Room_Type 1')
        input_data['market_segment_type'] = data.get('market_segment_type', 'Online')

        print(f"İşlenmiş veriler: {input_data}")

        # DataFrame oluştur
        input_df = pd.DataFrame([input_data])

        # Tahmin yap
        prediction = model.predict(input_df)[0]

        # Negatif fiyat olmaz
        if prediction < 0:
            prediction = 0

        # Sonucu döndür
        result = {
            'prediction': round(prediction, 2),
            'input_data': input_data
        }

        print(f"Tahmin sonucu: {result['prediction']}")
        return jsonify(result)

    except Exception as e:
        print(f"Tahmin hatası: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'message': 'Tahmin sırasında bir hata oluştu. Lütfen girdiğiniz değerleri kontrol edin.'
        }), 500


# Analiz sayfası
@app.route('/analytics')
def analytics():
    return render_template('analytics.html', model_info=model_info)


# Örnek veri API'si
@app.route('/api/sample_data')
def sample_data():
    try:
        df = pd.read_csv('Hotel_Reservations.csv', nrows=5)
        return jsonify(df.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)