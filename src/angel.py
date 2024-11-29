import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Configuración inicial
data_dir = "sample_data"  # Cambia esto a la ruta donde están tus carpetas "yes" y "no"
categories = ["yes", "no"]
img_size = 128

# 2. Función para cargar y preprocesar el dataset
def load_data():
    data = []
    for category in categories:
        folder = os.path.join(data_dir, category)
        label = categories.index(category)  # 0 para "no", 1 para "yes"
        for img in os.listdir(folder):
            try:
                img_path = os.path.join(folder, img)
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                resized_array = cv2.resize(img_array, (img_size, img_size))
                data.append([resized_array, label])
            except Exception as e:
                print(f"Error al procesar la imagen {img}: {e}")
    return data

# 3. Cargar y dividir el dataset
dataset = load_data()
np.random.shuffle(dataset)

X = np.array([item[0] for item in dataset]).reshape(-1, img_size, img_size, 1) / 255.0
y = np.array([item[1] for item in dataset])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Crear la CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Salida binaria
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Aumentación de datos
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
datagen.fit(X_train)

# 6. Entrenamiento del modelo
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_test, y_test),
                    epochs=20)

# 7. Evaluación del modelo
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy en entrenamiento: {train_acc}")
print(f"Accuracy en prueba: {test_acc}")

# 8. Matriz de confusión
y_pred = (model.predict(X_test) > 0.5).astype("int32")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# 9. Curvas de entrenamiento y validación
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Curva de Accuracy')
plt.show()

plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Curva de Pérdida')
plt.show()

# 10. Análisis del modelo
print("Clasificación:")
print(classification_report(y_test, y_pred, target_names=categories))
##############################################################################
# Curvas de pérdida (loss) durante el entrenamiento y la validación
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Pérdida - Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida - Validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Curva de Pérdida (Loss)')
plt.show()

# Curvas de precisión (accuracy) durante el entrenamiento y la validación
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Precisión - Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión - Validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.title('Curva de Precisión (Accuracy)')
plt.show()

# Comparación final entre los valores finales de entrenamiento y validación
train_loss = history.history['loss'][-1]
val_loss = history.history['val_loss'][-1]
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]

print(f"Pérdida en entrenamiento: {train_loss:.4f}")
print(f"Pérdida en validación: {val_loss:.4f}")
print(f"Precisión en entrenamiento: {train_acc:.4f}")
print(f"Precisión en validación: {val_acc:.4f}")

# Decisión sobre overfitting
if train_loss < val_loss and (val_loss - train_loss) > 0.1:
    print("El modelo muestra signos de overfitting: pérdida de validación mayor que pérdida de entrenamiento.")
if train_acc > val_acc and (train_acc - val_acc) > 0.1:
    print("El modelo muestra signos de overfitting: precisión de entrenamiento mucho mayor que la de validación.")
else:
    print("No hay signos evidentes de overfitting.")