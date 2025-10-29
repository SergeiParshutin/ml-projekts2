import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import skops.io as sio

# Ielādēsim datus
#bank_df = pd.read_csv("data/dati.csv", index_col="id", nrows=1000) 
bank_df = pd.read_csv("data/dati.csv", index_col="id") 
bank_df = bank_df.drop(["CustomerId", "Surname"], axis=1) 
bank_df = bank_df.sample(frac=1)

# Sadalīsim uz apmācības un testēšanas kopām
X = bank_df.drop(["Exited"], axis=1)
y = bank_df.Exited

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# Noradām kategorijas un skaitliskās kolonnas
cat_col = [1, 2]
num_col = [0, 3, 4, 5, 6, 7, 8, 9]

# Skaitlisko kolonnu normalizācija
numerical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", MinMaxScaler())]
)

# Kategoriju kolonnu transformācija
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder()),
    ]
)

# Apvienojam datu pirmapstrādes soļus
preproc_pipe = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, num_col),
        ("cat", categorical_transformer, cat_col),
    ],
    remainder="passthrough",
)

# Labāko atribūtu atlase
KBest = SelectKBest(chi2, k="all")

# Random Forest Classifier
#model = RandomForestClassifier(n_estimators=75, random_state=125)
model = RandomForestClassifier(n_estimators=25, random_state=125)

# Apmācības pipeline
train_pipe = Pipeline(
    steps=[
        ("KBest", KBest),
        ("RFmodel", model),
    ]
)

# Apvienojam pipeline
complete_pipe = Pipeline(
    steps=[
        ("preprocessor", preproc_pipe),
        ("train", train_pipe),
    ]
)

# Palaižam apmācību
complete_pipe.fit(X_train, y_train)


## Novērtējam modeli
predictions = complete_pipe.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")

print("Precizitāte:", str(round(accuracy, 2) * 100) + "%", "F1:", round(f1, 2))


## Confusion Matrix Plot
predictions = complete_pipe.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=complete_pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=complete_pipe.classes_)
disp.plot()
plt.savefig("model_results.png", dpi=120)

## Ierakstam metrikas datnē
with open("metrics.txt", "w") as outfile:
    outfile.write(f"\nPrecizitāte = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}\n\n")
   

# Saglabājam modeli
sio.dump(complete_pipe, "bank_pipeline.skops")
