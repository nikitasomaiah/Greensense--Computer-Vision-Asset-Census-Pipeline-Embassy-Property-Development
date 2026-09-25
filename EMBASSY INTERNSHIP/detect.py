"""
detect.py — GreenSense Tree & Asset Census module
---------------------------------------------------
Detects and counts landscape assets (trees, palms, shrubs, hedges, flower
beds) in site photos using OWL-ViT, an open-vocabulary object detector.

WHY OWL-ViT (and not a custom-trained YOLO model)?
Embassy doesn't have a labeled dataset of "tree vs palm vs shrub" bounding
boxes for its own sites, and building one by hand is weeks of annotation
work. OWL-ViT is pretrained to detect *any* object described in plain text
("a coconut palm tree", "a flowering shrub"), providing usable detections
from day one using Embassy's SOP vocabulary as the prompt list.

CONFIDENCE THRESHOLD TRADE-OFFS (Audit Sensitivity Tuning):
  - Low Threshold (0.05 - 0.10): High Recall / Sensitivity. Useful for dense,
    overlapping foliage where trees background-occlude each other. Trade-off:
    Higher rate of False Positives (e.g., misidentifying background foliage).
  - Standard Threshold (0.12 - 0.15): Default balanced setting for Embassy site
    surveys. Minimizes false alarms while reliably catching visible trees and shrubs.
  - High Threshold (0.20+): High Precision. Minimizes False Positives. Trade-off:
    Higher rate of False Negatives / missed detections, especially for smaller
    ground covers or distant plants.

Usage:
    from detect import CensusDetector
    det = CensusDetector()
    result = det.count_assets("site_photo.jpg")
    print(result.counts)
    
    # CLI Batch Mode:
    # python detect.py --batch ./site_folder/ --zone "Manyata Block D Garden"
"""

import argparse
from dataclasses import dataclass, field
import glob
import os
from typing import Dict, List, Optional, Tuple, Union

from PIL import Image, ImageDraw, ImageOps
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

from utils.geotag import extract_photo_metadata

DEFAULT_QUERIES: List[str] = [
    "a tree",
    "a palm tree",
    "a flowering shrub",
    "a hedge",
    "a flower bed",
    "a lawn / grass area",
    "a pergola",
    "a garden pathway",
]

CONFIDENCE_THRESHOLD = 0.12  # OWL-ViT scores run lower than closed-set detectors


def classify_plant_health(crop_image: Image.Image) -> Dict[str, Union[str, float, bool]]:
    """
    Placeholder / Stub interface for Nikita's MobileNetV2 plant-health classifier model.

    In production, this crops each detected bounding box (tree / shrub / palm)
    and passes it through the MobileNetV2 model to detect condition (healthy,
    chlorotic, leaf spot, pest infestation, dead).

    Returns:
        Dict with status, health score, and stub flag.
    """
    # STUB IMPLEMENTATION: Mock output demonstrating downstream integration
    # Production hook: return mobilenet_model.predict(crop_image)
    return {
        "status": "Healthy",
        "health_score": 0.94,
        "is_stub": True,
    }


@dataclass
class CensusResult:
    image_path: str
    counts: Dict[str, int] = field(default_factory=dict)
    boxes: List[dict] = field(default_factory=list)  # [{label, score, box, health_stub}]
    annotated_image: Optional[Image.Image] = None
    suggested_zone: Optional[str] = None
    gps_coords: Optional[Tuple[float, float]] = None


class CensusDetector:
    def __init__(self, model_name: str = "google/owlvit-base-patch32", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = None
        self._model = None
        self.model_name = model_name

    def _lazy_load_model(self):
        """Lazy load model on demand to save memory until detection is requested."""
        if self._model is None:
            self.processor = OwlViTProcessor.from_pretrained(self.model_name)
            self._model = OwlViTForObjectDetection.from_pretrained(self.model_name).to(self.device)
            self.model = self._model
            self.model.eval()

    def count_assets(
        self,
        image_input: Union[str, Image.Image],
        queries: List[str] = None,
        threshold: float = CONFIDENCE_THRESHOLD,
        draw_boxes: bool = True,
        check_health: bool = False,
    ) -> CensusResult:
        """
        Detects and counts landscape assets in an image file path or PIL Image object.
        Applies image validation and respects EXIF orientation.
        """
        self._lazy_load_model()
        queries = queries or DEFAULT_QUERIES

        # 1. Input Validation & Image Loading
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise ValueError(f"Image path does not exist: '{image_input}'")
            try:
                raw_image = Image.open(image_input)
                raw_image.verify()  # Verify integrity
                raw_image = Image.open(image_input)  # Re-open after verify
            except Exception as e:
                raise ValueError(f"Corrupt or unreadable image file '{image_input}': {e}")
            image_path = image_input
        elif isinstance(image_input, Image.Image):
            raw_image = image_input
            image_path = "in_memory_upload.jpg"
        else:
            raise ValueError("image_input must be a file path string or PIL Image object.")

        # Respect EXIF rotation tags (e.g. smartphone camera orientation)
        image = ImageOps.exif_transpose(raw_image).convert("RGB")

        # Extract EXIF metadata
        meta = extract_photo_metadata(raw_image)
        gps_coords = meta.get("gps")
        suggested_zone = meta.get("suggested_zone")

        # 2. Open-Vocabulary Object Detection
        inputs = self.processor(text=[queries], images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        if hasattr(self.processor, "post_process_grounded_object_detection"):
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=threshold
            )[0]
        elif hasattr(self.processor, "post_process_object_detection"):
            results = self.processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=threshold
            )[0]
        elif hasattr(self.processor.image_processor, "post_process_object_detection"):
            results = self.processor.image_processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=threshold
            )[0]
        else:
            raise RuntimeError("No compatible post-processing method found on OwlViTProcessor.")

        counts: Dict[str, int] = {q: 0 for q in queries}
        box_records = []

        # 3. Post-Process Detections
        for score, label_idx, box in zip(results["scores"], results["labels"], results["boxes"]):
            label = queries[label_idx]
            counts[label] += 1
            box_coords = [round(v, 1) for v in box.tolist()]

            record = {
                "label": label,
                "score": round(float(score), 3),
                "box": box_coords,
            }

            # Optional Plant Health Classifier Stub step
            if check_health:
                x0, y0, x1, y1 = [int(v) for v in box_coords]
                crop = image.crop((max(0, x0), max(0, y0), min(image.width, x1), min(image.height, y1)))
                record["health"] = classify_plant_health(crop)

            box_records.append(record)

        annotated = self._draw(image.copy(), box_records) if draw_boxes else None

        return CensusResult(
            image_path=image_path,
            counts=counts,
            boxes=box_records,
            annotated_image=annotated,
            suggested_zone=suggested_zone,
            gps_coords=gps_coords,
        )

    def batch_process_folder(
        self,
        folder_path: str,
        queries: List[str] = None,
        threshold: float = CONFIDENCE_THRESHOLD,
        zone_override: Optional[str] = None,
        log_to_register: bool = True,
    ) -> List[CensusResult]:
        """
        Runs detection on all image files (.jpg, .jpeg, .png) in a directory
        and logs results to the asset register.
        """
        from asset_register import log_census

        if not os.path.isdir(folder_path):
            raise ValueError(f"Folder path does not exist: '{folder_path}'")

        image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(folder_path, ext)))

        image_paths = sorted(list(set(image_paths)))
        if not image_paths:
            print(f"No image files found in folder: '{folder_path}'")
            return []

        results = []
        for path in image_paths:
            try:
                res = self.count_assets(path, queries=queries, threshold=threshold)
                results.append(res)

                if log_to_register:
                    assigned_zone = zone_override or res.suggested_zone or "Unspecified Zone"
                    log_census(zone=assigned_zone, counts=res.counts, image_path=path)
                    print(f"Successfully processed and logged: {path} -> Zone: '{assigned_zone}'")

            except Exception as e:
                print(f"Skipping corrupt or invalid file '{path}': {e}")

        return results

    @staticmethod
    def _draw(image: Image.Image, box_records: List[dict]) -> Image.Image:
        draw = ImageDraw.Draw(image)
        for rec in box_records:
            x0, y0, x1, y1 = rec["box"]
            draw.rectangle([x0, y0, x1, y1], outline="lime", width=3)
            label_text = f'{rec["label"]} {rec["score"]}'
            if "health" in rec:
                label_text += f' ({rec["health"]["status"]})'
            draw.text((x0, max(0, y0 - 12)), label_text, fill="lime")
        return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GreenSense Tree & Asset Census Detector")
    parser.add_argument("--image", type=str, help="Path to single photo for census detection")
    parser.add_argument("--batch", type=str, help="Path to folder of photos for batch detection")
    parser.add_argument("--zone", type=str, default=None, help="Zone name override for batch logging")
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--check-health", action="store_true", help="Run plant health classifier stub")

    args = parser.parse_args()

    detector = CensusDetector()

    if args.batch:
        print(f"Starting batch process on folder: {args.batch}")
        results = detector.batch_process_folder(
            folder_path=args.batch,
            threshold=args.threshold,
            zone_override=args.zone,
        )
        print(f"Batch completed: {len(results)} images processed.")
    elif args.image:
        result = detector.count_assets(args.image, threshold=args.threshold, check_health=args.check_health)
        print("Counts:", result.counts)
        if result.suggested_zone:
            print("Extracted EXIF Zone:", result.suggested_zone)
        out_path = "annotated_" + os.path.basename(args.image)
        if result.annotated_image:
            result.annotated_image.save(out_path)
            print("Saved annotated image to:", out_path)
    else:
        parser.print_help()
