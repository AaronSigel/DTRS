import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, ZeroPadding1D
from keras import optimizers
from keras.callbacks import ModelCheckpoint

class_names = ['Грузовая', 'Пассажирская', 'Промежуточная', 'Участковая']

# Load data
data = pd.read_excel('data.xlsx')

# Separate features and labels
X = data.iloc[:, :17]
y = data.iloc[:, 17]

# Convert labels to numerical values
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

# Convert labels to one-hot encoding
num_classes = len(np.unique(y))
y = np.eye(num_classes)[y]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape data for CNN
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Define CNN model
model = Sequential()
model.add(Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(ZeroPadding1D(padding=3))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(ZeroPadding1D(padding=3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(ZeroPadding1D(padding=3))
model.add(Conv1D(256, 3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(ZeroPadding1D(padding=3))
model.add(Conv1D(512, 3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(ZeroPadding1D(padding=3))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
sgd = optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

filepath="weights\\weights-{val_accuracy:.3f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Train model
model.fit(X_train, y_train, batch_size=300, epochs=1000, validation_data=(X_test, y_test), verbose=1, callbacks=[checkpoint])

model.evaluate(X_test, y_test)

# Check that predicted_classes are within range of valid indices for class_names
predicted_classes = np.argmax(model.predict(X_train), axis=-1)
predicted_classes = np.clip(predicted_classes, 0, len(class_names)-1)

# Convert predicted_classes to class names
predicted_class_names = [class_names[i] for i in predicted_classes]

# Save predicted class names to .xlsx file
# create a new DataFrame with the predicted class names
predicted_df = pd.DataFrame({'Predicted Class Names': predicted_class_names})
# concatenate the predicted_df with the original data DataFrame
result = pd.concat([data, predicted_df], axis=1)
# save the result DataFrame to a new excel file
with pd.ExcelWriter('result.xlsx') as writer:
    result.to_excel(writer, index=False)

# Load new data for testing
new_data = pd.read_excel('data_2.xlsx')

# Separate features and labels
X_new = new_data.iloc[:, :17]
y_new = new_data.iloc[:, 17]

# Convert labels to numerical values
encoder.fit(y_new)
y_new = encoder.transform(y_new)

# Convert labels to one-hot encoding
y_new = np.eye(num_classes)[y_new]

# Scale and reshape data
X_new = scaler.transform(X_new)
X_new = np.reshape(X_new, (X_new.shape[0], X_new.shape[1], 1))

# Predict classes for new data
y_pred = model.predict(X_new)
y_pred_classes = np.argmax(y_pred, axis=1)

# Check that predicted_classes are within range of valid indices for class_names
predicted_classes = np.argmax(model.predict(X_new), axis=-1)
predicted_classes = np.clip(predicted_classes, 0, len(class_names)-1)

# Convert predicted_classes to class names
predicted_class_names = [class_names[i] for i in predicted_classes]

# Decode predicted classes
y_pred_names = encoder.inverse_transform(y_pred_classes)

# Print predicted classes
# pprint(predicted_class_names)

# Save predicted class names to .xlsx file
# create a new DataFrame with the predicted class names
predicted_df = pd.DataFrame({'Predicted Class Names': predicted_class_names})
# concatenate the predicted_df with the original data DataFrame
result = pd.concat([new_data, predicted_df], axis=1)
# save the result DataFrame to a new excel file
with pd.ExcelWriter('result_2.xlsx') as writer:
    result.to_excel(writer, index=False)

# Save the entire model and weights
model.save('model.h5')

# Save the weights to a file in readable format
weights = model.get_weights()
with open('model_weights.txt', 'w') as f:
    f.write(pprint(str(weights)))
