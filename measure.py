import sys
import serial
import cv2
import propar
import csv
import os
import threading
from datetime import datetime, timedelta

# ---- Configuration ----
path = r'C:\Users\nst\Desktop\Anas B24\25A_30lpm_const40V'
FC_PORT = 'COM31'
dmm_port = 'COM5'
ard_port = 'COM30'

# ---- Initialize serial devices ----
dmm_ser = serial.Serial(port=dmm_port, baudrate=115200, timeout=0.1)
ard_ser = serial.Serial(port=ard_port, baudrate=9600, timeout=0.1)

try:
    mfc_ser = propar.instrument(FC_PORT)
except:
    print("Cannot open serial port (Flow Controller)...")
    sys.exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    sys.exit()

# ---- Global variables for Arduino sync ----
start_time = None
sync_time = None

# ---- Functions ----
def read_dmm(ser):
    ser.flushInput()
    ser.flushOutput()
    ser.write(('QM' + '\r').encode('utf-8'))
    response = b''
    second_eol = False
    while True:
        c = ser.read(1)
        if c:
            response += c
            if c == b'\r':
                if second_eol:
                    break
                else:
                    second_eol = True
        else:
            break
    return response

def decode_dmm(response):
    if len(response) > 0:
        response_string = response.decode("utf-8")
        response_split = response_string.split('\r')
        if len(response_split) == 3:
            measurement_split = response_split[1].split(',')
            if len(measurement_split) >= 1:
                return float(measurement_split[0])
    return None

def read_ard(ser):
    ser.flushInput()
    ser.flushOutput()
    response = b''
    SOL = False
    while True:
        c = ser.read(1)
        if c == b'\n':
            SOL = True
        if SOL:
            response += c
            if c == b'\r':
                break
    return response

def decode_ard(response):
    if len(response) > 0:
        response_string = response.decode("utf-8").strip('\r\n')
        measurement_split = response_string.split(' ')
        if len(measurement_split) == 2:
            current = float(measurement_split[0])
            arduino_time = int(measurement_split[1])
            return current, arduino_time
    return None, None

def ard_sync():
    global start_time, sync_time
    s_bytes = read_ard(ard_ser)
    start_time = datetime.now()
    decoded_bytes = s_bytes.decode("utf-8").strip('\r\n').split(' ')
    sync_time = int(decoded_bytes[1])
    print(f"Arduino synchronized. Sync-Time (ms): {sync_time}")
    print(f"PC Start time: {start_time}")

def arduino_time_to_pc_time(arduino_timestamp_ms):
    delta_ms = arduino_timestamp_ms - sync_time
    return start_time + timedelta(milliseconds=delta_ms)

def write_csv(filename, rows):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        header = ['TIMESTAMP', 'CURRENT (A)', 'VOLTAGE (V)', 'FLOW (l/s)']
        writer.writerow(header)
        writer.writerows(rows)

# ---- Data lists ----
current_data = []
voltage_data = []
flow_data = []

stop_threads = False

# ---- Data collection threads ----
def collect_current():
    while not stop_threads:
        current_val, arduino_time = decode_ard(read_ard(ard_ser))
        if current_val is not None and arduino_time is not None:
            timestamp = arduino_time_to_pc_time(arduino_time)
            current_data.append((timestamp, current_val))

def collect_voltage():
    while not stop_threads:
        voltage = decode_dmm(read_dmm(dmm_ser))
        if voltage is not None:
            voltage_data.append((datetime.now(), voltage))

def collect_flow():
    while not stop_threads:
        flow = mfc_ser.readParameter(205)
        flow_data.append((datetime.now(), flow))

# ---- Helper: find nearest value ----
def find_nearest(measurements, target_time):
    measurements.sort()
    nearest = min(measurements, key=lambda x: abs(x[0] - target_time))
    return nearest[1]

# ---- Main program ----
def main():
    global stop_threads

    print("Press 'e' to start recording, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('e'):
            ard_sync()

            # Start data collection threads
            stop_threads = False
            t1 = threading.Thread(target=collect_current)
            t2 = threading.Thread(target=collect_voltage)
            t3 = threading.Thread(target=collect_flow)
            t1.start()
            t2.start()
            t3.start()

            start_datetime = datetime.now()
            start_str = start_datetime.strftime('%Y-%m-%d_%H-%M-%S')

            video_filename = os.path.join(path, f'{start_str}.avi')
            csv_filename = os.path.join(path, f'{start_str}.csv')

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

            frame_measurements = []

            print("Recording... Press 'q' to stop.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = datetime.now()
                out.write(frame)
                cv2.imshow('frame', frame)

                frame_measurements.append(timestamp)

                if cv2.waitKey(1) == ord('q'):
                    break

            out.release()

            # Stop data collection threads
            stop_threads = True
            t1.join()
            t2.join()
            t3.join()

            # Synchronize data to frame timestamps
            final_measurements = []
            for ts in frame_measurements:
                ts_str = ts.strftime('%Y-%m-%d %H:%M:%S.%f')
                current = find_nearest(current_data, ts)
                voltage = find_nearest(voltage_data, ts)
                flow = find_nearest(flow_data, ts)
                final_measurements.append([ts_str, current, voltage, flow])

            write_csv(csv_filename, final_measurements)
            print(f"Video saved as {video_filename}")
            print(f"Data saved as {csv_filename}")
            break

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    dmm_ser.close()
    ard_ser.close()

if __name__ == '__main__':
    main()
