from transformers import pipeline
from PIL import Image
import torch
from torchvision import transforms
import pytesseract
import cv2
import numpy as np

class ImageAnalyzer:
    def __init__(self):
        # Initialize image captioning model
        self.captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
        
        # Initialize object detection model
        self.object_detector = pipeline("object-detection", model="hustvl/yolos-base")
        
        # Initialize OCR engine
        self.ocr_engine = pytesseract
    
    def analyze_image(self, image_path):
        """Comprehensive analysis of image content."""
        image = Image.open(image_path)
        
        results = {
            "caption": self._generate_caption(image),
            "objects": self._detect_objects(image),
            "text": self._extract_text(image),
            # "colors": self._analyze_colors(image),
            "composition": self._analyze_composition(image)
        }
        
        return results
    
    def _generate_caption(self, image):
        """Generate a natural language description of the image."""
        caption = self.captioner(image)[0]['generated_text']
        return caption
    
    def _detect_objects(self, image):
        """Detect and locate objects in the image."""
        detections = self.object_detector(image)
        
        # Filter and format results
        objects = {}
        for detection in detections:
            label = detection['label']
            confidence = detection['score']
            if confidence > 0.5:  # Confidence threshold
                if label not in objects:
                    objects[label] = []
                objects[label].append(confidence)

        # Average confidence for multiple detections of same object
        return {k: f"{len(v)} instance(s) (avg conf: {sum(v)/len(v):.2f})" 
                for k, v in objects.items()}
    
    def _extract_text(self, image):
        """Extract any text visible in the image using OCR."""
        # Convert PIL Image to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Preprocess for better OCR
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        preprocessed = cv2.threshold(gray, 0, 255, 
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Extract text
        text = self.ocr_engine.image_to_string(preprocessed)
        return text.strip() if text.strip() else None
    
    def _analyze_colors(self, image):
        """Analyze dominant colors in the image."""
        # Convert image to numpy array and reshape
        img_array = np.array(image)
        pixels = img_array.reshape(-1, 3)
        
        # Use K-means clustering to find dominant colors
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(pixels)
        
        # Get the colors and their percentages
        colors = []
        for center in kmeans.cluster_centers_:
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(center[0]), int(center[1]), int(center[2]))
            percentage = (kmeans.labels_ == 
                        list(kmeans.cluster_centers_).index(center)).sum() / len(pixels)
            colors.append({
                'color': hex_color,
                'percentage': f"{percentage:.1%}"
            })
            
        return sorted(colors, key=lambda x: float(x['percentage'][:-1]), 
                     reverse=True)
    
    def _analyze_composition(self, image):
        """Analyze basic composition elements of the image."""
        width, height = image.size
        img_array = np.array(image)
        
        # Calculate brightness distribution
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        brightness_mean = gray.mean()
        brightness_std = gray.std()
        
        composition = {
            "dimensions": f"{width}x{height}",
            "aspect_ratio": f"{width/height:.2f}",
            "brightness": {
                "mean": f"{brightness_mean:.1f}/255",
                "variation": f"{brightness_std:.1f}"
            }
        }
        
        return composition

    def get_summary(self, image_path):
        """Generate a human-readable summary of the image analysis."""
        results = self.analyze_image(image_path)
        
        summary = [
            f"Description: {results['caption']}",
            "\nObjects detected:",
            *[f"- {obj}: {count}" for obj, count in results['objects'].items()]
        ]
        
        if results['text']:
            summary.append(f"\nText found in image: {results['text']}")
            
        # summary.append("\nDominant colors:")
        # for color in results['colors'][:3]:  # Top 3 colors
        #     summary.append(f"- {color['color']} ({color['percentage']})")
            
        comp = results['composition']
        summary.append(f"\nImage properties:")
        summary.append(f"- Dimensions: {comp['dimensions']}")
        summary.append(f"- Aspect ratio: {comp['aspect_ratio']}")
        summary.append(f"- Average brightness: {comp['brightness']['mean']}")
        
        return "\n".join(summary)