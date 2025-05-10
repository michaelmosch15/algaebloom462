from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

def build_dnn(input_dim, num_classes=2):
    model = Sequential([
        Dense(32, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dense(16, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model