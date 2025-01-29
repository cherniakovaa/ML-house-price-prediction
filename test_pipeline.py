import pytest
import pandas as pd
import numpy as np
import pickle

def test_data_shape():
    """
    Sprawdzamy, czy plik data_clean.csv istnieje, jest odczytywany i nie jest pusty.
    """
    df = pd.read_csv(
        r"C:\Users\czern\Downloads\project\data_clean.csv",
        sep=','
    )

    assert not df.empty, "Plik data_clean.csv jest pusty lub nie został znaleziony."
    assert df.shape[0] > 0 and df.shape[1] > 0, "Niepoprawny kształt zbioru danych (wiersze/kolumny)."

def test_models_predictions():
    """
    Sprawdzamy, czy wytrenowany model XGBoost oraz scaler ładują się poprawnie
    i generują przewidywania o poprawnej długości (równej X.shape[0]).
    """
    #  Wczytujemy scaler i model
    with open('scaler.pkl', 'rb') as f:
        loaded_scaler = pickle.load(f)
    with open('model_xgb.pkl', 'rb') as f:
        best_model = pickle.load(f)

    #  Wczytujemy plik CSV
    df = pd.read_csv(
        r"C:\Users\czern\Downloads\project\data_clean.csv",
        sep=','
    )

    #  Zmieniamy nazwy kolumn (tak samo jak w głównym kodzie)
    new_column_names = [
        'area', 'rooms', 'floor', 'total_floor', 'year_built', 'attic', 'dish_washer', 'fridge', 'furniture',
        'oven', 'stove', 'tv', 'washer', 'secure_doors_windows', 'videophone', 'monitoring',
        'closed_area', 'balcony', 'basement', 'parking_space', 'alarm_system', 'for_non_smokers',
        'anti_burglary_blinds', 'elevator', 'separate_kitchen', 'internet', 'cable_tv', 'telephone', 'air_conditioning',
        'for_students', 'utility_room', 'terrace', 'two_level', 'garden', 'b_type_apartment',
        'b_type_panel', 'b_type_infill', 'b_type_loft', 'b_type_row',
        'b_type_detached', 'b_type_tenement', 'b_mat_aerated_concrete',
        'b_mat_brick', 'b_mat_concreate', 'b_mat_hollow_block', 'b_mat_large_panel', 'b_mat_expanded_clay',
        'b_mat_other', 'b_mat_reinforced_concrete', 'b_mat_silicate_brick', 'b_mat_wood',
        'windows_aluminum', 'windows_plastic', 'windows_wooden', 'heating_boiler', 'heating_central', 'heating_electric',
        'heating_gas', 'heating_other', 'status_not_ready', 'status_ready', 'status_renovation',
        'district_Bemowo', 'district_Białołęka', 'district_Bielany', 'district_Centrum', 'district_Metro_Wilanowska',
        'district_Mokotów', 'district_Ochota', 'district_Praga_Południe', 'district_Praga_Północ', 'district_Rembertów',
        'district_Targówek', 'district_Ursus', 'district_Ursynów', 'district_Warszawa', 'district_Wawer', 'district_Wesoła',
        'district_Wilanów', 'district_Wola', 'district_Włochy', 'district_mazowieckie', 'district_Śródmieście',
        'district_Żoliborz', 'gross_price'
    ]
    df.columns = new_column_names

    #  Usuwamy kolumny
    df.drop(columns=[
        'b_mat_aerated_concrete',
        'b_mat_brick',
        'b_mat_concreate',
        'b_mat_hollow_block',
        'b_mat_large_panel',
        'b_mat_expanded_clay',
        'b_mat_other',
        'b_mat_reinforced_concrete',
        'b_mat_silicate_brick',
        'b_mat_wood',
        'secure_doors_windows',
        'for_non_smokers',
        'for_students',
        'anti_burglary_blinds',
        'telephone',
        'status_not_ready',
        'status_ready',
        'status_renovation'
    ], inplace=True)

    #  Usuwamy NaN
    df.dropna(inplace=True)

    #  Rozdzielamy X i y
    X = df.drop(columns=['gross_price'])
    y = df['gross_price']

    #  Skalujemy kolumny numeryczne
    columns_to_scale = ['area', 'rooms', 'floor', 'total_floor', 'year_built']
    X_scaled = X.copy()
    X_scaled[columns_to_scale] = loaded_scaler.transform(X_scaled[columns_to_scale])

    #  Generujemy przewidywania
    preds = best_model.predict(X_scaled)

    #  Sprawdzamy wyniki
    assert len(preds) == len(X), "Liczba przewidywań nie zgadza się z liczbą wierszy w X."
    assert not np.isnan(preds).any(), "W przewidywaniach znajdują się wartości NaN!"
    print("\n=== Test OK ===")
    print(f"Model wygenerował {len(preds)} przewidywań w złotych – wszystko poprawne.")
