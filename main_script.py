import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Load the dataset
data = pd.read_csv("nba_games.csv")

# Select relevant features
# trb: Total Rebounds
# ast: Assists
# tov: Turnovers
# stl: Steals
# blk: Blocks
# ortg: Offensive Rating
# drtg: Defensive Rating
# ts%: True Shooting Percentage
selected_features = ['home', 'trb', 'ast', 'tov', 'stl', 'blk', '+/-', 'ortg', 'drtg', 'ts%']
X = data[selected_features]
y = data['won']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the neural network model
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(16, activation='relu'),
    Dropout(0.5),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)

# Save the model and scaler
model.save('neural_network_model.h5')
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Generate and save the classification report
from sklearn.metrics import classification_report
y_pred = (model.predict(X_test) > 0.5).astype("int32")
report = classification_report(y_test, y_pred, output_dict=True)
with open('classification_report.pkl', 'wb') as file:
    pickle.dump(report, file)

print(classification_report(y_test, y_pred))
