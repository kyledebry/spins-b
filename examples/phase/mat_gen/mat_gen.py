import numpy as np

h = 4.135667516e-15  # Plank's constant (eV s)
c = 299792458  # Speed of light (m/s)

with open('n_tantalum_oxide.dat', 'r') as n_file:
    n_wavelength = [line.split(',') for line in n_file.readlines()]
    wavelength = np.array([pair[0] for pair in n_wavelength]).astype(float)
    n = np.array([pair[1] for pair in n_wavelength]).astype(float)
    energy = h * c / (wavelength * 1e-6)

    for i in range(len(energy)):
        print(round(energy[i], 6), end=',')

    print('\n\n\n')

    for i in range(len(n)):
        print(n[i], end=',')

print('\n\n\n\n\n\n')

with open('k_tantalum_oxide.dat', 'r') as k_file:
    k_wavelength = [line.split(',') for line in k_file.readlines()]
    wavelength = np.array([pair[0] for pair in k_wavelength]).astype(float)
    k = np.array([pair[1] for pair in k_wavelength]).astype(float)
    energy = h * c / (wavelength * 1e-6)

    for i in range(len(energy)):
        print(round(energy[i], 6), end=',')

    print('\n\n\n')

    for i in range(len(k)):
        print(k[i], end=',')
