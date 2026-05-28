import numpy as np
from PIL import Image
import os

# Configuration
image_width = 176 
image_height = 144
color_mode = 'RGB'
hex_file_path = 'hex_data.txt'
output_folder = 'C:/Users/nurul/Downloads'
output_filename = 'test_image.png'

def read_image_from_hex_file(file_path, width, height, mode):
    with open(file_path, 'r') as file:
        hex_data = file.read().replace(' ', '').replace('\n', '').strip()

    print("Image data received. Re-decoding with Little Endian...")

    raw_bytes = bytes.fromhex(hex_data)
    
    # CHANGE: Using '<u2' (Little Endian) instead of '>u2'
    raw_data = np.frombuffer(raw_bytes, dtype='<u2')
    # Masking and shifting
    r = (raw_data >> 11) & 0x1F
    g = (raw_data >> 5) & 0x3F
    b = raw_data & 0x1F

    # Scale to 8-bit
    r = (r * 255) // 31
    g = (g * 255) // 63
    b = (b * 255) // 31

    rgb888 = np.stack((r, g, b), axis=-1).astype(np.uint8)
    
    # If the image looks "scrambled" horizontally, we might need to 
    # check if the hex string was copied completely.
    try:
        return Image.fromarray(rgb888.reshape((height, width, 3)), mode)
    except ValueError:
        print("Error: Data size doesn't match 176x144. Check your copy-paste!")
        return None
    
def save_image(image, folder, filename):
    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, filename)
    image.save(save_path)
    print(f"Success! Image saved to: {save_path}")

# Run it
if os.path.exists(hex_file_path):
    image = read_image_from_hex_file(hex_file_path, image_width, image_height, color_mode)
    save_image(image, output_folder, output_filename)
    image.show()
else:
    print(f"Error: {hex_file_path} not found. Paste your HEX data into that file first!")