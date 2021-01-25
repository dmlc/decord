from decord import AudioReader
import sys

filename = sys.argv[1]
sr = int(sys.argv[2])
ar = AudioReader(filename, sample_rate = sr)
print(ar[0])
print(ar[:])
print(ar.size())
print(ar.duration())
print(ar.get_num_padding())
