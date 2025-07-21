from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/LR'
RESULT_FOLDER = 'static/results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load model once
device = torch.device('cpu')
model_path = 'models/RRDB_ESRGAN_x4.pth'
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path, weights_only=False), strict=True)
model.eval().to(device)

def process_image(filename):
    lr_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    base = os.path.splitext(filename)[0]
    result_path = os.path.join(RESULT_FOLDER, f'{base}_rlt.png')

    img = cv2.imread(lr_path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)

    cv2.imwrite(result_path, output)
    return result_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result_path = process_image(filename)
            return render_template('index.html', input_img=filepath, output_img=result_path)
    return render_template('index.html', input_img=None, output_img=None)

if __name__ == '__main__':
    app.run(debug=True)
