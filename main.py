import pandas as pd
import numpy as np
import ast
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from sklearn.metrics import confusion_matrix, classification_report
# import seaborn as sns
# import matplotlib.pyplot as plt

# Load train data
train_data = pd.read_csv("../grape_backup/data/train.csv")


def transform_normalize(arr):
    img = np.array(ast.literal_eval(arr), dtype=np.float32) / 255.0
    if img.ndim == 1:
        img = img.reshape((64, 64, 3))
    return img


# Preprocess train images
train_data['image'] = train_data['image'].apply(transform_normalize)
X_train = np.stack(train_data['image'].values)
y_train = train_data['label'].values

# Encode labels
classes = np.unique(y_train)
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
y_train_encoded = np.array([class_to_idx[label] for label in y_train])

# Compute class weights for training
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_encoded),
    y=y_train_encoded
)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

# Build CNN model
input_shape = X_train.shape[1:]

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# Train model
history = model.fit(
    datagen.flow(X_train, y_train_encoded, batch_size=32),
    epochs=15,
    class_weight=class_weight_dict,
)

# -------------------
# Load and preprocess test data
# test_data = pd.read_csv("data/test.csv")
# test_data['image'] = test_data['image'].apply(transform_normalize)
# X_test = np.stack(test_data['image'].values)
# y_test = test_data['label'].values
# y_test_encoded = np.array([class_to_idx[label] for label in y_test])
#
# # Predict on test data
# y_pred_probs = model.predict(X_test)
# y_pred = np.argmax(y_pred_probs, axis=1)
#
# # Confusion matrix and classification report
# cm = confusion_matrix(y_test_encoded, y_pred)
# print("Confusion Matrix:")
# print(cm)
#
# print("\nClassification Report:")
# print(classification_report(y_test_encoded, y_pred, target_names=classes))
#
# # Optional: plot confusion matrix heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()

# model.save('grape_leaf_spot_model.keras')
model.save("saved_model/")
