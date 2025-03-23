# Arav Sharma
# Ari Gebhardt
# Raymond Chi
 
# Import Libraries
import os
import sys
import time
import socket
import argparse
import configparser
import subprocess
import threading
import numpy as np
import matplotlib
import csv

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

path = os.path.abspath('../../')
if not path in sys.path:
    sys.path.append(path)
import mmwsdr

def plot_recv_sig(rxtd):
    """
    Function to plot recieved signal, symbol, and spectrum
    """
    nframe, nread = rxtd.shape

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    ax = axes[0]
    first_frame = rxtd[0, :]
    ax.plot(first_frame.real[:50], label='I (Real)')
    ax.plot(first_frame.imag[:50], label='Q (Imag)')
    ax.set_xlabel('sample index')
    ax.set_ylabel('Amplitude')
    ax.set_title('time domain sig')
    ax.legend()
    ax.grid(True)

    ax = axes[1]
    sampling = max(1, nread // 1000)
    ax.scatter(first_frame.real[::sampling], first_frame.imag[::sampling], alpha=0.5, s=5)
    ax.set_xlabel('I')
    ax.set_ylabel('Q')
    ax.set_title('IQ Constellation')
    ax.grid(True)
    ax.axis('equal')

    ax = axes[2]
    fft_result=np.fft.fftshift(np.fft.fft(first_frame))

    ax.plot(np.arange(-nread//2, nread//2), 20*np.log10(np.abs(fft_result)))
    ax.set_xlabel('freq bin')

    ax.set_ylabel('mag')
    ax.set_title('spec')
    ax.grid(True)

    print('variance')

    print(np.var(first_frame))

    plt.tight_layout()
    plt.show()


def main():
    """
    Main function
    """

    # Parameters
    file_id = 0  # file id
    nfft = 1024  # num of continuous samples per frames
    nskip = 1024  # num of samples to skip between frames
    nframe = 3  # num of frames
    issave = False  # save the received IQ samples
    isdebug = False  # print debug messages
    iscalibrated = True  # apply calibration parameters
    sc_min = -400  # min sub-carrier index
    sc_max = 400  # max sub-carrier index
    #tx_pwr = 12000  # transmit power
    file_id = 0

    node = socket.gethostname().split('.')[0]  # Find the local hostname

    # Create an argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq", type=float, default=60.48e9, help="Carrier frequency in Hz (i.e., 60.48e9)")

    # Specify physical angle of RX array and TX power of signal
    parser.add_argument("--ang", type=float, default=0, help="Array Angle")
    parser.add_argument("--pow", type=int, default=0, help="Radiated Power")
    args = parser.parse_args()

    alpha = args.ang
    tx_pwr = args.pow
    print(alpha)
    print(tx_pwr)

    # Create a configuration parser
    config = configparser.ConfigParser()
    config.read('../../config/sivers.ini')

    # Create the SDR objects
    sdr1 = mmwsdr.sdr.Sivers60GHz(config=config, node='srv1-in1', freq=args.freq,
                                  isdebug=isdebug, islocal=(node == 'srv1-in1'), iscalibrated=iscalibrated)

    sdr2 = mmwsdr.sdr.Sivers60GHz(config=config, node='srv1-in2', freq=args.freq,
                                  isdebug=isdebug, islocal=(node == 'srv1-in2'), iscalibrated=iscalibrated)

    # Create the XY table controllers. Load the default location.
    if config['srv1-in1']['table_name'] != None:
        xytable1 = mmwsdr.utils.XYTable(config['srv1-in1']['table_name'], isdebug=isdebug)
        xytable1.move(x=float(config['srv1-in1']['x']), y=float(config['srv1-in1']['y']),
                      angle=float(config['srv1-in1']['angle']))

    if config['srv1-in2']['table_name'] != None:
        xytable2 = mmwsdr.utils.XYTable(config['srv1-in2']['table_name'], isdebug=isdebug)
        xytable2.move(x=float(config['srv1-in2']['x']), y=float(config['srv1-in2']['y']),
                      angle=float(config['srv1-in2']['angle']))

    # Create a wide-band tx signal (Single Tone)
    txtd = mmwsdr.utils.waveform.onetone()

    sdr1.send(txtd*tx_pwr)

    # start thread to show video of xytable2 moving to desired angle
    t = threading.Thread(target=xytable2.video)
    t.start()

    # Move TX at the center facing at 0 deg
    xytable1.move(x=650, y=650, angle=0)

    data = []
    beams = np.linspace(1, 63, 63)

    xytable2.move(x=650, y=650, angle=float(alpha))
    time.sleep(2)
    
    # Iterate through all beams and record the gain profile
    for beam_index in beams:
        sdr2.beam_index = beam_index
        time.sleep(0.5)

        # Receive data
        rxtd = sdr2.recv(nfft, nskip, nframe)
        frame = rxtd[1,:]
        var = np.var(frame)
        data.append(var)

        OUTPUT_FILE = "beam_iq.csv"

        with open(OUTPUT_FILE, "a") as f: # write IQ data to csv
            writer = csv.writer(f)
            writer.writerow([alpha, tx_pwr, beam_index] + list(frame))

    print(data)

    # Delete the SDR object. Close the TCP connections.
    del sdr1, sdr2

    OUTPUT_FILE = "beam_profiles.csv"

    with open(OUTPUT_FILE, "a") as f: # write gain profile to csv
        writer = csv.writer(f)
        writer.writerow([alpha] + [tx_pwr] +  data)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
