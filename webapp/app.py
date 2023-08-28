from flask import Flask, render_template, request, jsonify, send_from_directory, make_response
import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.textpath as textpath
import matplotlib
import os
import uuid
import pickle

matplotlib.use('Agg')

app = Flask(__name__)

# Set directories to save generated images and DFT data
UPLOAD_FOLDER = 'static/generated'
DATA_FOLDER = 'static/data'
for folder in [UPLOAD_FOLDER, DATA_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATA_FOLDER'] = DATA_FOLDER


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
    """Helper function to get the rendered width of a text for a given font size."""
    t = textpath.TextPath((0,0), text, size=fontsize)
    return t.get_extents().width

def add_text_to_image(ax, image_dim, text=None, color='white', text_width_percent=80):
    """Adds specified text to the image at the center."""
    if text:
        target_width = text_width_percent / 100 * image_dim[1]

        # Initial font size and width estimate
        fontsize = 1
        text_width = get_text_width(text, fontsize)

        # Iterative process to adjust font size
        max_iterations = 10
        for _ in range(max_iterations):
            # Adjust font size proportionally
            fontsize *= target_width / text_width
            text_width = get_text_width(text, fontsize)/2
            # Print all variables for debugging
            print(f"fontsize: {fontsize}, text_width: {text_width}, target_width: {target_width}")
            # Stop if the text width is within the target width
            # if text_width >= target_width:
            #     break

        # Draw the final text
        center_x = image_dim[1] / 2
        center_y = image_dim[0] / 2
        ax.text(center_x, center_y, text, color=color, fontweight='bold',
                fontsize=fontsize, ha='center', va='center')

def generate_image_from_dft(dft_slices, X_DIM, Y_DIM, Y_SCALE, HEX_START, HEX_END):
    colors = map_to_color(dft_slices, HEX_START, HEX_END)
    image = reshape(colors, X_DIM, Y_DIM, Y_SCALE)
    return image

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_audio():
    print(request.form)
    if 'file' not in request.files:
        return 'No file part', 400

    MUSIC_FILE_NAME = request.files['file']
    if MUSIC_FILE_NAME.filename == '':
        return 'No selected file', 400

    X_DIM = 100
    Y_SCALE = 2
    Y_DIM = int(X_DIM//Y_SCALE)

    y, sr = load_audio(MUSIC_FILE_NAME)
    dft_slices = compute_dft_slices(y, sr, X_DIM, Y_DIM)

    # Save DFT slices for later retrieval
    data_filename = str(uuid.uuid4()) + ".pkl"
    with open(os.path.join(app.config['DATA_FOLDER'], data_filename), 'wb') as f:
        pickle.dump(dft_slices, f)

    return jsonify({"data_filename": data_filename})


@app.route('/generate_image', methods=['POST'])
def generate_image():
    data_filename = request.form['data_filename']
    with open(os.path.join(app.config['DATA_FOLDER'], data_filename), 'rb') as f:
        dft_slices = pickle.load(f)

    X_DIM = int(request.form['x_dim'])
    Y_SCALE = int(request.form['y_scale'])
    Y_DIM = int(X_DIM//Y_SCALE)
    HEX_START = request.form['hex_start']
    HEX_END = request.form['hex_end']
    ARTIST_NAME = request.form['artist_name']
    SONG_TITLE = request.form['song_title']
    TEXT_COLOR = request.form['text_color']
    TEXT_WIDTH_PERCENT = int(request.form['text_width_percent'])

    image = generate_image_from_dft(dft_slices, X_DIM, Y_DIM, Y_SCALE, HEX_START, HEX_END)
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')

    image_dim = image.shape
    add_text_to_image(ax, image_dim, ARTIST_NAME, TEXT_COLOR, TEXT_WIDTH_PERCENT)
    # Turn off the axis
    plt.axis('off')
    # Ensure a tight layout
    plt.tight_layout()

    # Overwrite previous image or create new image file
    filename = "generated.png"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return jsonify({"image_url": '/' + filepath})


@app.route('/static/<path:path>')
def send_static_file(path):
    return send_from_directory('static', path)

# Prevent caching of generated image
@app.route('/static/generated/generated.png')
def serve_image():
    response = send_from_directory('static/generated', 'generated.png')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    return response

if __name__ == '__main__':
    app.run(debug=True)
