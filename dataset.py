import pandas as pd

data = pd.read_csv("driverSVT.csv")

# Colonnes pertinentes
feature_cols = [
    'speed','acc_X','acc_Y','acc_Z','perclos',
    'euleranglerotatephone','lightlevel'
]

# Vérification des valeurs manquantes
missing_values = data[feature_cols].isnull().sum()
print("Valeurs manquantes par colonne :")
print(missing_values)

print(f"\nTotal valeurs manquantes dans ces colonnes : {missing_values.sum()}")

from sklearn.impute import SimpleImputer
#remplissage

num_imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(num_imputer.fit_transform(data[feature_cols]), columns=feature_cols)

# Vérification après remplissage
print("\nValeurs manquantes après remplissage :")
print(X.isnull().sum())

#fin du verification
#training the model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Cible
y = data['dangerousstate']  # variable à prédire
le = LabelEncoder()
y = le.fit_transform(y)  # encode en 0,1,2 si multi-classe


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier

# Initialiser le modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraîner le modèle
model.fit(X_train, y_train)

def predict_driver_type(model, new_data, label_encoder):
    

    feature_cols = [
        'speed','acc_X','acc_Y','acc_Z','perclos',
        'euleranglerotatephone','lightlevel'
    ]

    # Vérifier que toutes les colonnes sont présentes
    for col in feature_cols:
        if col not in new_data.columns:
            raise ValueError(f"La colonne {col} est manquante dans les nouvelles données.")

    # Remplissage des valeurs manquantes par la médiane
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_new = pd.DataFrame(imputer.fit_transform(new_data[feature_cols]), columns=feature_cols)

    # Prédiction (nombres)
    predictions_num = model.predict(X_new)

    # Conversion en labels
    predictions_labels = label_encoder.inverse_transform(predictions_num)

    return predictions_labels