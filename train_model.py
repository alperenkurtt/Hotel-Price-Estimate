import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Dosya yolları oluştur
if not os.path.exists('models'):
    os.makedirs('models')

if not os.path.exists('static/images'):
    os.makedirs('static/images')


def train_and_save_models():
    print("Hotel fiyat tahmin modeli eğitimi başlıyor...")

    # Veri setini yükle
    df = pd.read_csv('Hotel_Reservations.csv')
    print(f"Veri seti yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")

    # booking_status'ü sayısal değere dönüştür
    df['booking_status'] = df['booking_status'].map({'Not_Canceled': 0, 'Canceled': 1})

    # Özellikler ve hedef değişken
    X = df.drop(columns=['avg_price_per_room', 'Booking_ID'])
    y = df['avg_price_per_room'].values

    print("Hedef değişken: avg_price_per_room")
    print("Özellik sayısı:", X.shape[1])

    # Kategorik ve sayısal sütunları belirle
    categorical_features = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
    numerical_features = [col for col in X.columns if col not in categorical_features]

    print(f"Kategorik özellikler: {categorical_features}")
    print(f"Sayısal özellikler: {numerical_features}")

    # Veriyi böl
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    print("Veri eğitim ve test setlerine bölündü")

    # Önişleme pipeline'ı oluştur
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ])

    # Farklı modelleri test et
    models = {
        "Linear Regression": LinearRegression(),
        "Support Vector Regression": SVR(kernel='rbf'),
        "Polynomial Regression": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
        "KNN Regression": KNeighborsRegressor(n_neighbors=5),
        "Decision Tree Regression": DecisionTreeRegressor(random_state=42),
        "Random Forest Regression": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    results = {}
    best_model = None
    best_score = -1
    best_pipeline = None

    print("\nModel performans sonuçları:")
    print("-" * 60)

    for name, model in models.items():
        # Pipeline oluştur
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        try:
            # Modeli eğit
            pipeline.fit(X_train, y_train)

            # Tahmin yap
            y_pred = pipeline.predict(X_test)

            # Metrikleri hesapla
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            results[name] = {
                "MSE": mse,
                "RMSE": rmse,
                "R2": r2,
                "MAE": mae
            }

            print(f"{name:<25} R²: {r2:.4f} | RMSE: {rmse:.2f} | MAE: {mae:.2f}")

            # En iyi modeli bul
            if r2 > best_score:
                best_score = r2
                best_model = name
                best_pipeline = pipeline

        except Exception as e:
            print(f"{name}: Hata - {str(e)}")
            results[name] = {"Error": str(e)}

    print(f"\nEn iyi model: {best_model} (R² = {best_score:.4f})")

    # En iyi modeli kaydet
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(best_pipeline, f)

    # Model bilgilerini kaydet
    model_info = {
        'model_name': best_model,
        'r2': results[best_model]['R2'],
        'rmse': results[best_model]['RMSE'],
        'mae': results[best_model]['MAE'],
        'mse': results[best_model]['MSE'],
        'all_results': results,
        'categorical_features': categorical_features,
        'numerical_features': numerical_features
    }

    with open('models/model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)

    # Benzersiz değerleri kaydet (form için gerekli)
    unique_values = {
        'meal_plans': df['type_of_meal_plan'].unique().tolist(),
        'room_types': df['room_type_reserved'].unique().tolist(),
        'market_segments': df['market_segment_type'].unique().tolist()
    }

    with open('models/unique_values.pkl', 'wb') as f:
        pickle.dump(unique_values, f)

    # Görselleştirmeler oluştur
    create_visualizations(df, results, y_test, best_pipeline.predict(X_test))

    print("Model eğitimi ve kaydı tamamlandı!")
    return model_info


def create_visualizations(df, results, y_test, y_pred):
    print("Görselleştirmeler oluşturuluyor...")

    # 1. Model performans karşılaştırması
    plt.figure(figsize=(12, 6))
    model_names = []
    r2_scores = []

    for name, metrics in results.items():
        if 'R2' in metrics:
            model_names.append(name)
            r2_scores.append(metrics['R2'])

    bars = plt.bar(model_names, r2_scores, color='skyblue')
    plt.title('Model Performans Karşılaştırması (R² Skoru)')
    plt.ylabel('R² Skoru')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)

    # En iyi modeli vurgula
    max_index = r2_scores.index(max(r2_scores))
    bars[max_index].set_color('green')

    # Değerleri yazalım
    for i, v in enumerate(r2_scores):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center')

    plt.tight_layout()
    plt.savefig('static/images/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Fiyat dağılımı
    plt.figure(figsize=(10, 6))
    plt.hist(df['avg_price_per_room'], bins=50, color='lightblue', edgecolor='black', alpha=0.7)
    plt.title('Hotel Oda Fiyatları Dağılımı')
    plt.xlabel('Ortalama Oda Fiyatı')
    plt.ylabel('Frekans')
    plt.grid(True, alpha=0.3)
    plt.savefig('static/images/price_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Gerçek vs Tahmin grafiği
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Gerçek Fiyat')
    plt.ylabel('Tahmin Edilen Fiyat')
    plt.title('Gerçek vs Tahmin Edilen Fiyatlar')
    plt.grid(True, alpha=0.3)
    plt.savefig('static/images/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Oda türüne göre ortalama fiyat
    plt.figure(figsize=(12, 6))
    room_prices = df.groupby('room_type_reserved')['avg_price_per_room'].mean().sort_values(ascending=False)
    room_prices.plot(kind='bar', color='coral')
    plt.title('Oda Türüne Göre Ortalama Fiyat')
    plt.xlabel('Oda Türü')
    plt.ylabel('Ortalama Fiyat')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('static/images/room_price_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Görselleştirmeler tamamlandı")