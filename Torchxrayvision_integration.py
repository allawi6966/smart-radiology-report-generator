"""
TorchXRayVision Integration Module
Provides image-grounded classifiers for verifying radiological findings.
"""

import torch
import torchvision
import torchxrayvision as xrv
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class TorchXRayVisionVerifier:
    """
    Wrapper around TorchXRayVision models for verifying radiological findings.
    Uses pre-trained models to assess evidence for claimed findings.
    """
    
    def __init__(self, model_name: str = "densenet121-res224-all"):
        """
        Initialize TorchXRayVision verifier.
        
        Args:
            model_name: Pre-trained model to use. Options:
                - "densenet121-res224-all" (default, most comprehensive)
                - "densenet121-res224-pc" (pneumonia/consolidation focused)
                - "resnet50-res512-all"
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[TXV] Using device: {self.device}")
        
        # Load pre-trained model
        print(f"[TXV] Loading model: {model_name}")
        self.model = xrv.models.DenseNet(weights=model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Store class names
        self.classes = xrv.datasets.default_pathologies
        print(f"[TXV] Loaded {len(self.classes)} pathology classes")
        print(f"[TXV] Classes: {self.classes}")
        
        # Figure out the square resolution this model expects (e.g. 224, 512)
        # from the "-resNNN-" segment in the weights name, defaulting to 224.
        self.resolution = 224
        for part in model_name.split('-'):
            if part.startswith('res') and part[3:].isdigit():
                self.resolution = int(part[3:])
                break
        
        # torchxrayvision models require a square input image. This is the
        # standard preprocessing pipeline from the torchxrayvision docs:
        # center-crop to a square, then resize to the model's resolution.
        self.transform = torchvision.transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(self.resolution),
        ])
        
    def preprocess_image(self, image_path_or_array) -> torch.Tensor:
        """
        Preprocess image to model input format.
        
        Args:
            image_path_or_array: Path to image file or numpy array
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Load image if path provided
        if isinstance(image_path_or_array, str):
            img = Image.open(image_path_or_array).convert('L')  # Grayscale
            img_array = np.array(img)
        else:
            img_array = image_path_or_array
        
        # Ensure uint8
        if img_array.dtype != np.uint8:
            max_val = img_array.max() if img_array.max() > 0 else 1
            img_array = (img_array / max_val * 255).astype(np.uint8)
        
        # Normalize to the [-1024, 1024] range TorchXRayVision models expect
        img_norm = xrv.datasets.normalize(img_array, 255)
        
        # Add channel dim -> [1, H, W]
        img_norm = img_norm[None, ...]
        
        # Center-crop to square + resize to the model's expected resolution.
        # This is required: the model raises if H != W.
        img_norm = self.transform(img_norm)
        
        img_tensor = torch.from_numpy(img_norm).float().unsqueeze(0)  # [1, 1, H, W]
        
        return img_tensor.to(self.device)
    
    def extract_region(self, image_array: np.ndarray, bbox: Dict) -> np.ndarray:
        """
        Extract region of interest from image using bounding box.
        
        Args:
            image_array: Input image as numpy array
            bbox: Bounding box dict with keys: x, y, width, height (normalized 0-1)
            
        Returns:
            Extracted region as numpy array
        """
        h, w = image_array.shape[:2]
        
        # Convert normalized coords to pixel coords
        x_px = int(bbox['x'] * w)
        y_px = int(bbox['y'] * h)
        w_px = int(bbox['width'] * w)
        h_px = int(bbox['height'] * h)
        
        # Ensure bounds
        x_px = max(0, min(x_px, w - 1))
        y_px = max(0, min(y_px, h - 1))
        w_px = min(w_px, w - x_px)
        h_px = min(h_px, h - y_px)
        
        # Guard against degenerate/too-small crops (e.g. a near-zero-area
        # bbox) which don't carry enough signal for the pathology model and
        # can break the center-crop/resize step downstream.
        min_size = 32
        if w_px < min_size or h_px < min_size:
            return image_array
        
        return image_array[y_px:y_px + h_px, x_px:x_px + w_px]
    
    def get_pathology_scores(self, image_tensor: torch.Tensor) -> Dict[str, float]:
        """
        Get pathology detection scores for full image.
        
        Args:
            image_tensor: Preprocessed image tensor [1, 1, H, W]
            
        Returns:
            Dict mapping pathology name to detection score (0-1)
        """
        with torch.no_grad():
            logits = self.model(image_tensor)
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        scores = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
        return scores
    
    def verify_finding(self, 
                      image_path_or_array,
                      finding_name: str,
                      finding_description: str,
                      bbox: Optional[Dict] = None,
                      use_region: bool = True) -> Dict:
        """
        Verify if a claimed finding is supported by the image.
        
        Args:
            image_path_or_array: Path to image or numpy array
            finding_name: Name of the finding (e.g., "Right Upper Lobe Opacity")
            finding_description: Description of the finding
            bbox: Bounding box of the finding (optional)
            use_region: If True and bbox provided, analyze region. Otherwise analyze full image.
            
        Returns:
            Dict with verification results:
            {
                'finding_name': str,
                'supported': bool,
                'evidence_score': float (0-1),
                'matching_pathologies': Dict[str, float],
                'top_pathologies': List[Tuple[str, float]],
                'recommendation': str
            }
        """
        # Load image
        if isinstance(image_path_or_array, str):
            img = Image.open(image_path_or_array).convert('L')
            img_array = np.array(img)
        else:
            img_array = image_path_or_array.copy()
        
        # Decide which region to analyze
        if use_region and bbox is not None:
            analysis_region = self.extract_region(img_array, bbox)
            region_type = "bounding box region"
        else:
            analysis_region = img_array
            region_type = "full image"
        
        # Preprocess and get scores
        img_tensor = self.preprocess_image(analysis_region)
        pathology_scores = self.get_pathology_scores(img_tensor)
        
        # Map finding to relevant pathologies
        relevant_pathologies = self._map_finding_to_pathologies(finding_name, finding_description)
        
        # Calculate evidence score from relevant pathologies
        matching_scores = {path: score for path, score in pathology_scores.items() 
                          if path in relevant_pathologies}
        
        if matching_scores:
            evidence_score = max(matching_scores.values())
            max_pathology = max(matching_scores, key=matching_scores.get)
        else:
            evidence_score = 0.0
            max_pathology = None
        
        # Rank all pathologies
        sorted_pathologies = sorted(pathology_scores.items(), key=lambda x: x[1], reverse=True)
        top_pathologies = sorted_pathologies[:5]  # Top 5
        
        # Decision logic
        supported = evidence_score >= 0.4  # Threshold
        
        if supported:
            recommendation = f"✓ SUPPORTED: Strong evidence of {max_pathology} (score: {evidence_score:.2f})"
        elif evidence_score >= 0.25:
            recommendation = f"⚠ UNCERTAIN: Weak evidence. Max score: {max_pathology} ({evidence_score:.2f})"
        else:
            recommendation = f"✗ UNSUPPORTED: No clear evidence. Image shows: {top_pathologies[0][0]}"
        
        return {
            'finding_name': finding_name,
            'region_analyzed': region_type,
            'supported': supported,
            'evidence_score': evidence_score,
            'matching_pathologies': matching_scores,
            'top_pathologies': top_pathologies,
            'recommendation': recommendation,
            'all_scores': pathology_scores
        }
    
    def _map_finding_to_pathologies(self, finding_name: str, description: str) -> List[str]:
        """
        Map a clinical finding name to relevant TorchXRayVision pathologies.
        This is a heuristic mapping; can be improved with domain knowledge.
        """
        mapping = {
            'opacity': ['Consolidation', 'Pneumonia', 'Infiltration', 'Atelectasis'],
            'consolidation': ['Consolidation', 'Pneumonia', 'Infiltration'],
            'pneumonia': ['Pneumonia', 'Consolidation', 'Infiltration'],
            'infiltrate': ['Infiltration', 'Consolidation', 'Pneumonia'],
            'effusion': ['Pleural Effusion', 'Consolidation'],
            'nodule': ['Nodule', 'Infiltration'],
            'mass': ['Mass', 'Nodule'],
            'atelectasis': ['Atelectasis', 'Consolidation'],
            'emphysema': ['Emphysema'],
            'pneumothorax': ['Pneumothorax'],
            'edema': ['Edema', 'Consolidation', 'Infiltration'],
        }
        
        # Search in finding name and description
        search_text = (finding_name + " " + description).lower()
        
        relevant = set()
        for keyword, pathologies in mapping.items():
            if keyword in search_text:
                relevant.update(pathologies)
        
        # If no specific mapping found, return common ones
        if not relevant:
            relevant = {'Consolidation', 'Infiltration', 'Pneumonia', 'Atelectasis'}
        
        return list(relevant)


class RegionAnalyzer:
    """Utility for analyzing specific image regions."""
    
    @staticmethod
    def visualize_with_bbox(image_path: str, bbox: Dict, output_path: str):
        """
        Draw bounding box on image and save.
        
        Args:
            image_path: Path to input image
            bbox: Bounding box dict
            output_path: Path to save annotated image
        """
        from PIL import ImageDraw
        
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        h, w = img.size[1], img.size[0]
        x_px = int(bbox['x'] * w)
        y_px = int(bbox['y'] * h)
        w_px = int(bbox['width'] * w)
        h_px = int(bbox['height'] * h)
        
        # Draw box
        draw.rectangle([x_px, y_px, x_px + w_px, y_px + h_px], 
                      outline='red', width=3)
        img.save(output_path)
        print(f"[RegionAnalyzer] Saved annotated image to {output_path}")


if __name__ == "__main__":
    print("TorchXRayVision Integration Module")
    print("Use this module in finding_verifier.py")