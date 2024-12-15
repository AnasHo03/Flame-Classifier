import cv2
import time
import csv
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from LSTMClassifier import LSTMClassifierTrain
from CNNInference import CNNInference

# Variablen initialisieren
measuring = True
measurements = []
lstm_outputs_argmax = []
lstm_outputs_filtered = []
start_time = None
current_measure_value = 0
color_mapping = {0: (0, 255, 0), 1: (0, 255, 255), 2: (0, 0, 255)}
model_id = r"runs\segment\train23\weights\best.pt"
checkpoint = r"lightning_logs\novar_600ep_200h_2l\checkpoints\epoch=599-step=180000.ckpt"

# Tensor-Sequenz initialisieren
sequence_length = 100
tensor_sequence = torch.zeros(sequence_length, 4)

cap = cv2.VideoCapture(0)

CNNmodel = CNNInference(model_id)

lstm_model = LSTMClassifierTrain.load_from_checkpoint(checkpoint)
lstm_model.to("cuda")

def save_measurements_to_csv(measurements, filename="measurements.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time (s)", "Predicted State"])
        writer.writerows(measurements)

def plot_measurements(measurements, lstm_outputs_weighted):
    # Zeiten und Werte aus den Messungen extrahieren
    times = [measurement[0] for measurement in measurements]
    values = [measurement[1] for measurement in measurements]
    
    # Zeiten und gewichtete LSTM-Ausgaben extrahieren
    times_weighted = [output[0] for output in lstm_outputs_weighted]
    weighted_values = [output[1] for output in lstm_outputs_weighted]

    # Erstelle den Plot
    plt.figure(figsize=(12, 6))
    
    # Beobachtete Zustände (Observed State) plotten
    plt.plot(times, values, marker='o', label='Observed State')
    
    # Gewichtete LSTM-Ausgaben (Predicted State) plotten
    plt.plot(times_weighted, weighted_values, marker='x', label='Predicted State')
    
    # Setze y-Ticks manuell auf die Werte 0, 1, und 2 ohne Nachkommastellen
    plt.yticks([0, 1, 2], ['0', '1', '2'])
    
    # Achsenbeschriftungen und Titel
    plt.xlabel('Time, s')
    plt.ylabel('State')
    plt.title('Observed and LSTM Outputs Over Time')
    
    # Legende und Gitter aktivieren
    plt.legend()
    plt.grid(True)
    
    # Layout verbessern und Plot anzeigen
    plt.tight_layout()
    plt.show()

def update_tensor_sequence(tensor_sequence, new_tensor):
    tensor_sequence = torch.cat((tensor_sequence[1:], new_tensor.unsqueeze(0)), dim=0)
    return tensor_sequence

def tiefpassfilter(timed_sequence, time_constant):
    filtered_sequence = []
    sequence = [val for time, val in timed_sequence]
    for i in range(len(sequence)):
        # Berechne den Durchschnitt der letzten `time_constant` Werte
        if i < time_constant:
            average_value = np.mean(sequence[:i+1])
        else:
            average_value = np.mean(sequence[i-time_constant+1:i+1])
        
        # Runde den Durchschnittswert auf den nächsten ganzzahligen Zustand (0, 1, 2)
        filtered_value = int(round(average_value))
        
        # Da wir nur 0, 1 oder 2 als gültige Zustände haben, stellen wir sicher,
        # dass der gerundete Wert innerhalb dieser Grenzen bleibt
        filtered_value = max(0, min(2, filtered_value))
        
        filtered_sequence.append(filtered_value)
    
    return filtered_sequence

source = input("Bitte geben Sie 'webcam' für die Webcam oder den Pfad zu einer Videodatei ein: ").strip()
if source.lower() == 'webcam':
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(source)

if not cap.isOpened():
    print("Fehler beim Zugriff auf die Kamera oder Videodatei")
    exit()
elapsed_time = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Fehler beim Zugriff auf die Kamera")
        break

    if measuring:
        # Bilderkennung durchführen
        CNNmodel.detect(frame, conf=0.5)
        in1 = torch.tensor(CNNmodel.dis_x_list, dtype=torch.float32).view(-1, 1)
        in2 = torch.tensor(CNNmodel.dis_y_list, dtype=torch.float32).view(-1, 1)
        in3 = torch.tensor(CNNmodel.la_len_list, dtype=torch.float32).view(-1, 1)
        in4 = torch.tensor(CNNmodel.ua_len_list, dtype=torch.float32).view(-1, 1)
        inputs = torch.cat([in1, in2, in3, in4], dim=1).squeeze(0)
        
        # Tensor-Sequenz aktualisieren
        tensor_sequence = update_tensor_sequence(tensor_sequence, inputs)

        # LSTM ausführen
        sequence_tensor = tensor_sequence.unsqueeze(0).to("cuda")
        lstm_output = lstm_model(sequence_tensor)
        
        # Softmax auf den LSTM-Output anwenden
        softmax_output = torch.softmax(lstm_output, dim=1).squeeze(0)
        
        # Argmax und gewichtete Summe der Softmax-Einträge berechnen
        elapsed_time += 0.040734
        argmax_value = torch.argmax(softmax_output).item()
        weighted_value = torch.sum(softmax_output * torch.arange(len(softmax_output), dtype=torch.float32).to("cuda")).item()
        
        lstm_outputs_argmax.append((elapsed_time, argmax_value))

        measurements.append((elapsed_time, current_measure_value))
        cv2.circle(frame, (50, 50), 20, color_mapping[current_measure_value], -1)

    cv2.imshow('Webcam', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('e') and not measuring:
        measuring = True
        start_time = time.time()
        print("Messung gestartet.")
        
    elif key == ord('q') and measuring:
        measuring = False
        print("Messung gestoppt.")
        save_measurements_to_csv(measurements)
        
    elif key == ord('1'):
        current_measure_value = 0
        print("Messwert geändert zu 1.")
        
    elif key == ord('2'):
        current_measure_value = 1
        print("Messwert geändert zu 2.")
        
    elif key == ord('3'):
        current_measure_value = 2
        print("Messwert geändert zu 3.")
        
    elif key == ord('q') and not measuring:
        break

# Ressourcen freigeben
cap.release()
cv2.destroyAllWindows()
#%%
filtered_values = tiefpassfilter(lstm_outputs_argmax, 10)
lstm_outputs_filtered = [(lstm_outputs_argmax[i][0], filtered_values[i]) for i in range(len(filtered_values))]
# Messwerte plotten
plot_measurements(measurements, lstm_outputs_filtered)
times = [measurement[0] for measurement in measurements]
save_measurements_to_csv(zip(times, filtered_values), filename="predicted_values.txt")