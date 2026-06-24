import torch
print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')

from faster_whisper import WhisperModel
print('faster-whisper: OK')

import sounddevice
print('sounddevice: OK')

import soundfile
print('soundfile: OK')

from kokoro import KPipeline
print('kokoro: OK')

import silero_vad
print('silero-vad: OK')

print('\nAll imports successful!')
