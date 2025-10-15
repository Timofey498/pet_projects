import numpy as np

def decode_barker_message(signal: np.ndarray) -> str:
    barker = [+1, +1, +1, -1, -1, -1, +1, -1, -1, +1, -1]
    signal_norm = signal /np.max(signal)
    expanded_barker = np.repeat(barker, 5)
    expanded_barker_inv = [-x for x in expanded_barker]
    bit_data = (signal_norm > 0.5).astype(int)
    cor = np.correlate(bit_data, expanded_barker, mode = 'valid')
    begin_mes = np.argmax(cor) + 55
    bit_data = bit_data[begin_mes:]
    grouped = bit_data[:len(bit_data) - len(bit_data) % 5].reshape(-1, 5)
    bits = (grouped.mean(axis=1) > 0.5).astype(int)
    byts = bits[:len(bits) - len(bits)%8].reshape(-1, 8)
    bit_strings = [''.join(map(str, b)) for b in byts]
    ascii_codes = [int(s, 2) for s in bit_strings]
    message = ''.join(map(chr, ascii_codes))
    return message
    



