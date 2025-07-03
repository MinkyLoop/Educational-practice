from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import logging
import os
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='static')
CORS(app) 

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLASS_NAMES = [

    'airport_inside', 'artstudio', 'auditorium', 'bakery', 'bar', 'bathroom', 'bedroom', 'bookstore',
    'bowling', 'buffet', 'casino', 'children_room', 'church_inside', 'classroom', 'cloister',
    'closet', 'clothingstore', 'computerroom', 'concert_hall', 'corridor', 'deli', 'dentaloffice',
    'dining_room', 'elevator', 'fastfood_restaurant', 'florist', 'gameroom', 'garage', 'greenhouse',
    'grocerystore', 'gym', 'hairsalon', 'hospitalroom', 'inside_bus', 'inside_subway', 'jewelleryshop',
    'kindergarden', 'kitchen', 'laboratorywet', 'laundromat', 'library', 'livingroom', 'lobby',
    'locker_room', 'mall', 'meeting_room', 'movietheater', 'museum', 'nursery', 'office',
    'operating_room', 'pantry', 'poolinside', 'prisoncell', 'restaurant', 'restaurant_kitchen',
    'shoeshop', 'stairscase', 'studiomusic', 'subway', 'toystore', 'trainstation',
    'tv_studio', 'videostore', 'waitingroom', 'warehouse', 'winecellar'

]

def load_model():
    try:
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Используется устройство: {device}")
        
        
        model = torch.load('model', map_location=device)
        model.eval()
        
        logger.info("Модель успешно загружена")

        return model, device
    
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {str(e)}")
        raise

model, device = load_model()

transform = transforms.Compose([

    transforms.Resize((256, 256)),
    transforms.ToTensor()

])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():

    if 'image' not in request.files: return jsonify({'error': 'No image part', 'status': 'error'}), 400
    
    file = request.files['image']
    
    if file.filename == '': return jsonify({'error': 'No selected file', 'status': 'error'}), 400
    
    if not allowed_file(file.filename): return jsonify({'error': 'Invalid file type', 'status': 'error'}), 400
    
    try:

        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(temp_path)
        
        image = Image.open(temp_path).convert('RGB')
        image = transform(image)
        
        def to_device(data, device):
            
            if isinstance(data, (list,tuple)): return [to_device(x, device) for x in data]

            return data.to(device, non_blocking=True)
        
        def predict_image(img, model):

            xb = to_device(img.unsqueeze(0), device)
            yb = model(xb)
            prob, preds = torch.max(yb, dim=1)

            return CLASS_NAMES[preds[0].item()], float(prob.item())
        
        with torch.no_grad(): predicted_class, confidence = predict_image(image, model)
        
        result = {

            'predicted_class': predicted_class,
            'status': 'success'

        }
        
        os.remove(temp_path)
        
        return jsonify(result)
        
    except Exception as e:

        logger.error(f"Ошибка предсказания: {str(e)}")
        
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/')
def serve_index(): return send_from_directory('static', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path): return send_from_directory('static', path)

if __name__ == '__main__': app.run(host='0.0.0.0', port=5000, debug=True)