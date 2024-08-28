import torch
import torchvision.transforms as transforms
from PIL import Image

def process_image(image):
    """
    Process an input PIL Image for prediction.
    
    Args:
        image (PIL.Image): Input image
    
    Returns:
        torch.Tensor: Processed image tensor
    """
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)
    return image

def predict(model, image):
    """
    Make a prediction using the provided model and image.
    
    Args:
        model (torch.nn.Module): Trained PyTorch model
        image (torch.Tensor): Processed image tensor
    
    Returns:
        int: Predicted class index
    """
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

def load_image(image_file):
    """
    Load an image file and convert it to RGB mode.
    
    Args:
        image_file (file-like object): Uploaded image file
    
    Returns:
        PIL.Image: Loaded image in RGB mode
    """
    image = Image.open(image_file)
    image = image.convert('RGB')
    return image

def get_prediction(model, image_file):
    """
    Process an image file and return the prediction.
    
    Args:
        model (torch.nn.Module): Trained PyTorch model
        image_file (file-like object): Uploaded image file
    
    Returns:
        int: Predicted class index
    """
    image = load_image(image_file)
    processed_image = process_image(image)
    prediction = predict(model, processed_image)
    return prediction