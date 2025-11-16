import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Charger le dataset
data = pd.read_csv("driverSVT.csv",nrows=500000)

# Colonnes utiles
feature_cols = [
    'speed','acc_X','acc_Y','acc_Z','perclos','lightlevel',
    'euleranglerotatephone_roll','euleranglerotatephone_pitch','euleranglerotatephone_yaw'
]

# Vérification des valeurs manquantes
missing_values = data[feature_cols].isnull().sum()
print("Valeurs manquantes :\n", missing_values)
print("\nTotal :", missing_values.sum())

# Remplissage (par médiane)
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(data[feature_cols]), columns=feature_cols)

print("\nAprès remplissage :")
print(X.isnull().sum())

# Cible
y = data['dangerousstate']
le = LabelEncoder()
y = le.fit_transform(y)

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

# Fonction de prédiction
def predict_driver_type(model, new_data, label_encoder):

    # Vérifier colonnes
    for col in feature_cols:
        if col not in new_data.columns:
            raise ValueError(f"Colonne manquante : {col}")

    # Remplir valeurs manquantes
    imputer = SimpleImputer(strategy='median')
    X_new = pd.DataFrame(imputer.fit_transform(new_data[feature_cols]),
                         columns=feature_cols)

    # Prédiction de la classe (numérique)
    pred_num = model.predict(X_new)

    # Conversion en label d'origine
    pred_label = label_encoder.inverse_transform(pred_num)

    return pred_label



