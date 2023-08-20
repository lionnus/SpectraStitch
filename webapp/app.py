from flask import Flask, render_template, request, send_file
import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io
import textwrap

app = Flask(__name__)

def load_audio(file):
    y, sr = librosa.load(file, sr=None)
    return y, sr

def compute_dft_slices(y, sr, X_DIM, Y_DIM):
    n_slices = X_DIM * Y_DIM
    slice_len = len(y) // n_slices
    slices = [y[i:i+slice_len] for i in range(0, slice_len * n_slices, slice_len)]
    dft_slices = [np.abs(np.fft.fft(slice)) for slice in slices]
    return dft_slices

def create_colormap(start_hex, end_hex):
    start_rgb = mcolors.hex2color(start_hex)
    end_rgb = mcolors.hex2color(end_hex)
    return mcolors.LinearSegmentedColormap.from_list("custom_colormap", [start_rgb, end_rgb], N=256)

def map_to_color(dft_slices, start_hex="#000000", end_hex="#FFFFFF"):
    max_frequencies = [np.argmax(slice) for slice in dft_slices]
    normalized_frequencies = np.array(max_frequencies) / max(max_frequencies)
    colormap = create_colormap(start_hex, end_hex)
    colors = colormap(normalized_frequencies)
    return colors

def reshape(colors, X_DIM, Y_DIM, Y_SCALE):
    image = np.reshape(colors, (Y_DIM, X_DIM, 4))[:, :, :3]
    new_image = []
    for row in image:
        for _ in range(Y_SCALE):
            new_image.append(row)
    return np.array(new_image)

def get_text_width(text, fontsize):
    fig_temp = plt.figure(figsize=(10, 10), dpi=80)
    ax_temp = fig_temp.add_subplot(111)
    t = ax_temp.text(0.5, 0.5, text, fontsize=fontsize, visible=True, transform=ax_temp.transAxes)
    fig_temp.canvas.draw()
    width = t.get_window_extent().width
    plt.close(fig_temp)
    return width

def add_text_to_image(ax, image_dim, text=None, color='white', text_width_percent=80):
    if text:
        target_width = text_width_percent / 100 * image_dim[1]
        chars_per_line = len(text) * target_width / image_dim[1]
        wrapped_text = '\n'.join(textwrap.wrap(text, width=int(chars_per_line)))
        fontsize = 12
        text_width = get_text_width(wrapped_text, fontsize)
        max_iterations = 10
        for _ in range(max_iterations):
            fontsize *= target_width / text_width
            text_width = get_text_width(wrapped_text, fontsize)
        center_x = image_dim[1] / 2
        center_y = image_dim[0] / 2
        ax.text(center_x, center_y, wrapped_text, color=color, fontweight='bold', fontsize=fontsize, ha='center', va='center')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_audio():
    MUSIC_FILE_NAME = request.files['file']
    X_DIM = int(request.form['x_dim'])
    Y_DIM = int(request.form['y_dim'])
    Y_SCALE = int(request.form['y_scale'])
    HEX_START = request.form['hex_start']
    HEX_END = request.form['hex_end']
    PDF_RESOLUTION = int(request.form['pdf_resolution'])
    text = request.form['text']
    text_color = request.form['text_color']
    text_width_percent = int(request.form['text_width_percent'])

    y, sr = load_audio(MUSIC_FILE_NAME)
    dft_slices = compute_dft_slices(y, sr, X_DIM, Y_DIM)
    colors = map_to_color(dft_slices, HEX_START, HEX_END)
    image = reshape(colors, X_DIM, Y_DIM, Y_SCALE)

    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')

    image_dim = image.shape
    add_text_to_image(ax, image_dim, text, text_color, text_width_percent)
    
    output = io.BytesIO()
    plt.savefig(output, format='PNG', dpi=PDF_RESOLUTION)
    plt.close(fig)
    output.seek(0)

    return send_file(output, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
