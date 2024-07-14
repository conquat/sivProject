import librosa
import numpy as np
import matplotlib.pyplot as plt
import PIL
from progressbar import progressbar

SOUND_SPEED = 343  # m/s
CENTER_MICROPHONE = np.array([0.0, 0.0, 0.0])
TARGET_GRID_CENTER = np.array([0.0, 0.0, 5.0])
IMAGE_DIM = (30, 30)
mic_array_dim = (5, 5)
mic_spacing = (0.3, 0.3)
scan_plane_size = (8.0, 8.0)

SAMPLE_RATE = 22050
violin = librosa.load('audio/violin_c.wav')[0]
hz440 = librosa.load('audio/hz440.wav')[0]
noise = librosa.load('audio/noise.wav')[0]
hz100 = librosa.load('audio/100hz-107657.mp3')[0]
hz1000 = librosa.load('audio/beep-sound-of-1000-hz-95208.mp3')[0]


def get_delay(a, b):
    return np.linalg.norm(a - b) / SOUND_SPEED

def freq_to_index(freq, spectrum_size, SAMPLE_RATE):
    return int(freq / SAMPLE_RATE * spectrum_size)

def hermite(a):
    return np.conj(np.transpose(a))


microphone_positions = []
for i in range(mic_array_dim[0]):
    for j in range(mic_array_dim[1]):
        microphone_positions.append(CENTER_MICROPHONE + np.array([
            (j - np.floor(mic_array_dim[0] / 2)) * mic_spacing[0],
            (i - np.floor(mic_array_dim[1] / 2)) * mic_spacing[1],
            0.0
        ]))

test_points = []
for i in range(-int(IMAGE_DIM[1] / 2), int((IMAGE_DIM[1] + 1) / 2)):
    for j in range(-int(IMAGE_DIM[0] / 2), int((IMAGE_DIM[0] + 1) / 2)):
        test_points.append(TARGET_GRID_CENTER + np.array([
            scan_plane_size[0] / (IMAGE_DIM[0] - 1) * j,
            scan_plane_size[1] / (IMAGE_DIM[1] - 1) * i,
            0.0
        ]))
        
        
def get_spectrums(sources, microphone_positions, start_index, sample_length):
    spectrums = []
    for mic_pos in microphone_positions:
        sounds = []
        for position, track in sources:
            time_delay = get_delay(TARGET_GRID_CENTER + np.append(position, 0.0), mic_pos)
            index_delay = int(time_delay * SAMPLE_RATE)
            sounds.append(track[start_index + index_delay:start_index + index_delay + sample_length])
        recorded_sound = sum(sounds)
        coeff = np.sqrt(np.average(recorded_sound**2)) * 10.0
        noise = (np.random.random((len(recorded_sound))) - 0.5) * coeff
        spectrum = np.fft.fft(recorded_sound + noise) / (4 * np.pi * np.linalg.norm(mic_pos - TARGET_GRID_CENTER))
        spectrums.append(spectrum)
    return spectrums

sources = [
    (np.array([2.0, -2.0]), hz1000),
    (np.array([-2.0, -2.0]), hz440),
]

sample_duration = 0.2
sample_length = int(sample_duration * SAMPLE_RATE)
start_time = 0.0
start_index = int(start_time * SAMPLE_RATE)


spectrums = get_spectrums(sources, microphone_positions, start_index, sample_length)


test_freq = 1000
freq_index = freq_to_index(test_freq, sample_length, SAMPLE_RATE)
test_freq_intensisies = np.array([spectrum[freq_index] for spectrum in spectrums]).reshape(len(spectrums), 1)
csm = test_freq_intensisies @ np.conj(np.transpose(test_freq_intensisies))

def get_steering_vector_component(mic_pos, test_pos, frequency):
    distance = np.linalg.norm(mic_pos - test_pos)
    delay = get_delay(test_pos, mic_pos)
    omega = 2 * np.pi * frequency
    return 4 * np.pi * distance * np.exp(-1j * omega * delay)

def get_steering_vector(test_pos, microphone_positions, freq):
    steering_vector = np.array(
        [
            get_steering_vector_component(mic_pos, test_pos, freq)
            for mic_pos in microphone_positions
        ]) / np.linalg.norm(test_pos - CENTER_MICROPHONE)
    return steering_vector.reshape(len(microphone_positions), 1)

def get_steering_vectors(test_points, microphone_positions, freq):
    steering_vectors = np.zeros((len(test_points), len(microphone_positions)), dtype=np.complex_)
    for i, test_point in enumerate(test_points):
        steering_vector = get_steering_vector(test_point, microphone_positions, freq)
        steering_vectors[i] = steering_vector.reshape((len(microphone_positions)))
    return steering_vectors.reshape(len(test_points), len(microphone_positions), 1)
    
steering_vectors = get_steering_vectors(test_points, microphone_positions, test_freq)



def get_Y_mat(steering_vectors, csm):
    mat = np.zeros((len(steering_vectors)), dtype=np.complex_)
    mic_count = len(csm)
    for i, steering_vector in enumerate(progressbar(steering_vectors)):
        mat[i] = np.take(hermite(steering_vector) @ csm @ steering_vector, 0) / mic_count**2
    return mat

def get_A_mat(steering_vectors):
    inv_steering_vectors = np.reciprocal(steering_vectors)
    mat = np.zeros((len(steering_vectors), len(steering_vectors)), dtype=np.complex_)
    csvm_mods = np.conjugate(inv_steering_vectors) @ inv_steering_vectors.reshape(len(inv_steering_vectors), 1, len(microphone_positions))
    for i, steering_vector in enumerate(progressbar(steering_vectors)):
        for j, csvm_mod in enumerate(csvm_mods):
            mat[i, j] = np.take(hermite(steering_vector) @ csvm_mod @ steering_vector, 0) / len(csvm_mod)**2
    return mat

print("creating a mat")
a_mat = get_A_mat(steering_vectors)
print("creating y mat")
y_mat = get_Y_mat(steering_vectors, csm)


x_mat = np.zeros((len(steering_vectors)), dtype=np.complex_)

def iteration(n):
    sum_1 = 0
    for k in range(0, n):
        f1 = a_mat[n, k]
        f2 = x_mat[k]
        sum_1 += f1 * f2

    sum_2 = 0
    for k in range(n+1, len(x_mat)):
        f1 = a_mat[n, k]
        f2 = x_mat[k]
        sum_2 += f1 * f2
    x_mat[n] = max(y_mat[n] - (np.take(sum_1, 0) + np.take(sum_2, 0)), 0)

for l in progressbar(range(10)):
    for n in range(len(x_mat)):
        iteration(n)
    for n in range(len(x_mat) - 1, -1, -1):
        iteration(n)

def show_mat(mat):
    im = PIL.Image.new(mode="RGB", size=IMAGE_DIM, color=(0, 0, 0))
    mat_abs = abs(mat)
    norm = 255 / max(mat_abs)
    for i in range(IMAGE_DIM[1]):
        for j in range(IMAGE_DIM[0]):
            val = mat_abs[i * IMAGE_DIM[1] + j] * norm
            im.putpixel((j, i), (abs(int(val.real * 0)), abs(int(val.imag * 0)), int(val)))
    im.show()

show_mat(x_mat)
show_mat(y_mat)
