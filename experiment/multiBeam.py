import subprocess
import numpy as np

def run_beam_scans():
    pows = np.logspace(1,4,5,dtype=int)
    for angle in range(-32, 46):
        for power in pows:
            # Run experiments as subprocesses with different angle and tx power
            # Is more stable when running large simulations
            command = ["python", "IQprofile.py", "--ang", str(angle), "--pow", str(power)]
            print("Running:", " ".join(command))
            subprocess.call(command)

if __name__=="__main__":
    run_beam_scans()
