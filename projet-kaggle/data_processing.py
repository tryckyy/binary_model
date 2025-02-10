import os
import pandas as pd
import numpy as np
from joblib import delayed, Parallel

def load_and_preprocess_data(train_file, test_file, processed_train_file, processed_test_file):
    if os.path.exists(processed_train_file) and os.path.exists(processed_test_file):
        train = pd.read_csv(processed_train_file)
        test = pd.read_csv(processed_test_file)
        print(f'Train shape: {train.shape}')
        print(f'Test shape: {test.shape}')
        print("Données chargées depuis les fichiers prétraités.")
    else:
        test = pd.read_csv(test_file).drop('id', axis=1)
        train = pd.read_csv(train_file).drop('id', axis=1)

        print(f'Train shape: {train.shape}')
        print(f'Test shape: {test.shape}')

        missing_threshold = 0.5
        missing_train = train.isnull().mean()
        columns_to_drop = [col for col in missing_train[missing_train > missing_threshold].index]
        train.drop(columns=columns_to_drop, inplace=True)
        test.drop(columns=columns_to_drop, inplace=True)

        print(f'Columns dropped: {columns_to_drop}')

        numerical_cols = train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = train.select_dtypes(include=[object]).columns.tolist()

        train[numerical_cols] = train[numerical_cols].fillna(train[numerical_cols].median())
        test[numerical_cols] = test[numerical_cols].fillna(train[numerical_cols].median())

        categorical_cols_in_test = [col for col in categorical_cols if col in test.columns]

        train[categorical_cols_in_test] = train[categorical_cols_in_test].fillna('not_present')
        test[categorical_cols_in_test] = test[categorical_cols_in_test].fillna('not_present')

        min_vals = train[numerical_cols].min()
        max_vals = train[numerical_cols].max()
        train[numerical_cols] = (train[numerical_cols] - min_vals) / (max_vals - min_vals)
        test[numerical_cols] = (test[numerical_cols] - min_vals) / (max_vals - min_vals)

        train[numerical_cols] = train[numerical_cols].apply(lambda x: np.maximum(x, 1e-9))
        test[numerical_cols] = test[numerical_cols].apply(lambda x: np.maximum(x, 1e-9))

        train[numerical_cols] = np.log1p(train[numerical_cols])
        test[numerical_cols] = np.log1p(test[numerical_cols])

    categorical_cols = train.select_dtypes(include=[object]).columns.tolist()

    def process_column(col, train, test):
        if col in train.columns:
            # Calculer les fréquences des catégories dans train et test
            freq_train = train[col].value_counts(normalize=True)
            freq_train.name = 'freq_train'

            freq_test = test[col].value_counts(normalize=True) if col in test.columns else pd.Series()
            freq_test.name = 'freq_test'

            freq = pd.merge(freq_train, freq_test, how='outer', left_index=True, right_index=True)
            freq['max'] = freq.max(axis=1)

            rare_categories = freq[freq['max'] < 0.01].index

            if len(rare_categories) > 0:
                train[col] = train[col].replace(rare_categories, 'infrequent')
                if col in test.columns:
                    test[col] = test[col].replace(rare_categories, 'infrequent')

            print(f"Colonne '{col}': {len(rare_categories)} catégories rares remplacées.")

    def replace(train, test, categorical_cols):
        Parallel(n_jobs=-1)(delayed(process_column)(col, train, test) for col in categorical_cols)

    replace(train, test, categorical_cols)

    for col in categorical_cols:
        if col in train.columns:
            unique_categories = np.unique(train[col])
            category_to_int = {category: idx for idx, category in enumerate(unique_categories)}

            train[col] = train[col].map(category_to_int)
            if col in test.columns:
                test[col] = test[col].map(category_to_int)

            print(f"Colonne '{col}' encodée.")

    train.to_csv(processed_train_file, index=False)
    test.to_csv(processed_test_file, index=False)
    print(f"Données sauvegardées dans {processed_train_file} et {processed_test_file}")

    return train, test
