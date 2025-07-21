import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

model_path = 'models\\RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cpu')  # change to 'cuda' if using GPU

test_img_folder = 'LR/*'

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path, weights_only=False), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

idx = 0
for path in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(idx, base)
    
    # Read and preprocess input
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0).to(device)

    # Run model
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    
    # Postprocess output
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # Convert to HWC BGR
    output = (output * 255.0).round().astype(np.uint8)  # Convert to uint8

    # Save result
    cv2.imwrite('results/{:s}_rlt.png'.format(base), output)




# # # Using Diffrenet Method 
# import os
# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# from flask import Flask, request, jsonify, send_file
# import RRDBNet_arch as arch  # Ensure this module is available

# app = Flask(__name__)

# # Update the PyTorch model path
# MODEL_PATH = "models/RRDB_ESRGAN_x4.pth"

# # Check if the model file exists
# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

# # Load the ESRGAN model
# device = torch.device("cpu")  # Change to 'cuda' if using GPU
# model = arch.RRDBNet(3, 3, 64, 23, gc=32)  # Ensure architecture matches the trained model

# # Load model weights
# state_dict = torch.load(MODEL_PATH, map_location=device)
# model.load_state_dict(state_dict, strict=True)
# model.eval().to(device)

# # Function to preprocess input image
# def preprocess_image(image_path):
#     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
#     img = np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))  # Convert BGR to RGB
#     img = torch.from_numpy(img).float().unsqueeze(0).to(device)  # Add batch dimension
#     return img

# # Function to postprocess the output
# def postprocess_image(output):
#     output = output.squeeze().cpu().detach().numpy()  # Convert to NumPy
#     output = np.transpose(output, (1, 2, 0))  # Convert back to HWC format
#     output = np.clip(output * 255.0, 0, 255).astype(np.uint8)  # Scale to 0-255
#     return output

# @app.route("/", methods=["GET"])
# def home():
#     return "Welcome to RFDB ESRGAN Super-Resolution API!"

# @app.route("/super-resolve", methods=["POST"])
# def super_resolve():
#     if "image" not in request.files:
#         return jsonify({"error": "No image uploaded"}), 400

#     image_file = request.files["image"]
#     input_path = f"uploads/{image_file.filename}"
#     output_path = f"results/{image_file.filename.split('.')[0]}_enhanced.png"

#     # Ensure directories exist
#     os.makedirs("uploads", exist_ok=True)
#     os.makedirs("results", exist_ok=True)

#     # Save input image
#     image_file.save(input_path)

#     # Preprocess the image
#     img_input = preprocess_image(input_path)

#     # Run inference
#     with torch.no_grad():
#         output = model(img_input).cpu().detach().squeeze()

#     # Postprocess and save the output image
#     enhanced_img = postprocess_image(output)
#     cv2.imwrite(output_path, enhanced_img)

#     return send_file(output_path, mimetype="image/png", as_attachment=True)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)


