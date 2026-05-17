import os
import joblib
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

emotion_map = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear"
}

def get_emotion_from_filename(filename):
    parts = filename.split("-")
    if len(parts) < 3:
        return None
    emotion_code = parts[2]
    return emotion_map.get(emotion_code)

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_dataset(dataset_path):
    X = []
    y = []

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                emotion = get_emotion_from_filename(file)

                if emotion is None:
                    continue

                file_path = os.path.join(root, file)
                features = extract_features(file_path)

                if features is not None:
                    X.append(features)
                    y.append(emotion)

    return np.array(X), np.array(y)

dataset_path = "dataset" 

X, y = load_dataset(dataset_path)

print("Features shape:", X.shape)
print("Labels shape:", y.shape)
print("Unique emotions:", set(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ՀԻՄՆական՝ ԱՎԵԼԱՑՎԱԾ Է probability=True ՊԱՐԱՄԵՏՐԸ
model = SVC(kernel="rbf", C=5.0, gamma='scale', probability=True) 
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, 'emotion_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nԱվարտվեց: Մոդելը և Ստանդարտիզատորը հաջողությամբ պահպանվեցին:")
print("✅ Մոդելը հիմա ունի predict_proba() հնարավորությունը")
