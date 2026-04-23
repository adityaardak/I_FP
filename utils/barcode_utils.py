from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps


@dataclass(slots=True)
class DetectedRegion:
    bbox: tuple[int, int, int, int]
    padded_bbox: tuple[int, int, int, int]
    confidence: float
    crop: Image.Image
    crop_id: str


@dataclass(slots=True)
class DecodedValue:
    value: str
    symbology: str
    bbox: tuple[int, int, int, int] | None = None
    confidence: float | None = None
    decoder: str | None = None
    preprocessing: str | None = None
    rotation: int = 0
    source_region_id: str | None = None


@dataclass(slots=True)
class BarcodeScanResult:
    annotated_image: Image.Image
    regions: list[DetectedRegion]
    decoded_values: list[DecodedValue]
    decoder_trace: list[str] = field(default_factory=list)
    failure_message: str | None = None
    suggestions: list[str] = field(default_factory=list)
    source_name: str | None = None

    @property
    def primary_decoded(self) -> DecodedValue | None:
        return self.decoded_values[0] if self.decoded_values else None


def _clean_text(value: Any) -> str:
    return str(value).strip()


def normalize_code(value: Any) -> str:
    return "".join(character.upper() for character in _clean_text(value) if character.isalnum())


def normalize_lookup_text(value: Any) -> str:
    return _clean_text(value)


@lru_cache(maxsize=1)
def load_barcode_detector():
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        from ultralytics import YOLO
    except Exception:
        return None

    repo_id = "Piero2411/YOLOV8s-Barcode-Detection"
    try:
        repo_files = list_repo_files(repo_id)
        preferred_filenames = [
            "best.pt",
            "weights/best.pt",
            "model.pt",
            "barcode.pt",
        ]
        model_file = next((name for name in preferred_filenames if name in repo_files), None)
        if model_file is None:
            model_file = next((name for name in repo_files if name.lower().endswith(".pt")), None)
        if model_file is None:
            return None
        model_path = hf_hub_download(repo_id=repo_id, filename=model_file)
        return YOLO(model_path)
    except Exception:
        return None


def get_barcode_decoder_status() -> dict[str, bool]:
    status = {
        "zxing-cpp": False,
        "opencv-barcode": hasattr(cv2, "barcode_BarcodeDetector") or (hasattr(cv2, "barcode") and hasattr(cv2.barcode, "BarcodeDetector")),
        "pyzbar": False,
        "qr-detector": True,
    }
    try:
        import zxingcpp  # noqa: F401

        status["zxing-cpp"] = True
    except Exception:
        status["zxing-cpp"] = False

    try:
        from pyzbar.pyzbar import decode as _decode  # noqa: F401

        status["pyzbar"] = True
    except Exception:
        status["pyzbar"] = False

    return status


def pad_barcode_bbox(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    horizontal_ratio: float = 0.2,
    vertical_ratio: float = 0.45,
    min_padding: int = 12,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    width, height = image_size
    box_width = max(1, right - left)
    box_height = max(1, bottom - top)

    pad_x = max(min_padding, int(box_width * horizontal_ratio))
    pad_y = max(min_padding, int(box_height * vertical_ratio))

    padded = (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(width, right + pad_x),
        min(height, bottom + pad_y),
    )
    return padded


def _safe_crop(image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    left, top, right, bottom = bbox
    width, height = image.size
    clipped = (
        max(0, left),
        max(0, top),
        min(width, right),
        min(height, bottom),
    )
    return image.crop(clipped)


def detect_regions(image: Image.Image, confidence_threshold: float = 0.25) -> list[DetectedRegion]:
    rgb_image = image.convert("RGB")
    detector = load_barcode_detector()
    if detector is None:
        return []

    bgr_image = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
    try:
        results = detector.predict(source=bgr_image, conf=confidence_threshold, verbose=False)
    except Exception:
        return []

    if not results:
        return []

    boxes = results[0].boxes
    if boxes is None:
        return []

    detected_regions: list[DetectedRegion] = []
    for index, (xyxy, confidence) in enumerate(zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()), start=1):
        bbox = tuple(int(value) for value in xyxy.tolist())
        padded_bbox = pad_barcode_bbox(bbox, rgb_image.size)
        detected_regions.append(
            DetectedRegion(
                bbox=bbox,
                padded_bbox=padded_bbox,
                confidence=float(confidence),
                crop=_safe_crop(rgb_image, padded_bbox),
                crop_id=f"crop-{index}",
            )
        )
    return detected_regions


def _add_white_margin(image_array: np.ndarray, min_border: int = 12) -> np.ndarray:
    border = max(min_border, int(min(image_array.shape[:2]) * 0.08))
    return cv2.copyMakeBorder(
        image_array,
        border,
        border,
        border,
        border,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255) if image_array.ndim == 3 else 255,
    )


def _deskew_image(image_array: np.ndarray) -> np.ndarray | None:
    gray = image_array if image_array.ndim == 2 else cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = 255 - thresholded
    points = cv2.findNonZero(inverted)
    if points is None or len(points) < 20:
        return None

    rect = cv2.minAreaRect(points)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 1 or abs(angle) > 20:
        return None

    center = (image_array.shape[1] // 2, image_array.shape[0] // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    border_value = (255, 255, 255) if image_array.ndim == 3 else 255
    rotated = cv2.warpAffine(
        image_array,
        matrix,
        (image_array.shape[1], image_array.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return rotated


def generate_preprocessing_variants(image: Image.Image) -> list[tuple[str, np.ndarray]]:
    rgb_array = np.array(ImageOps.exif_transpose(image.convert("RGB")))
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    equalized = cv2.equalizeHist(gray)
    upscaled_2x = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    upscaled_3x = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, otsu = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(equalized, -1, sharpen_kernel)
    inverted = cv2.bitwise_not(otsu)
    contrast = cv2.convertScaleAbs(equalized, alpha=1.7, beta=0)

    variants = [
        ("original", _add_white_margin(rgb_array)),
        ("grayscale", _add_white_margin(gray)),
        ("upscaled_2x", _add_white_margin(upscaled_2x)),
        ("upscaled_3x", _add_white_margin(upscaled_3x)),
        ("otsu_threshold", _add_white_margin(otsu)),
        ("adaptive_threshold", _add_white_margin(adaptive)),
        ("sharpened", _add_white_margin(sharpened)),
        ("inverted", _add_white_margin(inverted)),
        ("contrast_enhanced", _add_white_margin(contrast)),
    ]

    deskewed = _deskew_image(equalized)
    if deskewed is not None:
        variants.append(("deskewed", _add_white_margin(deskewed)))
    return variants


def _rotate_variant(image_array: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return image_array
    if angle == 90:
        return cv2.rotate(image_array, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image_array, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image_array


def _array_to_bbox(points: Any) -> tuple[int, int, int, int] | None:
    if points is None:
        return None
    point_array = np.array(points)
    if point_array.size == 0:
        return None
    point_array = point_array.reshape(-1, 2)
    x_values = point_array[:, 0]
    y_values = point_array[:, 1]
    return (
        int(np.min(x_values)),
        int(np.min(y_values)),
        int(np.max(x_values)),
        int(np.max(y_values)),
    )


def _decode_with_zxing(image_array: np.ndarray) -> list[DecodedValue]:
    try:
        import zxingcpp
    except Exception:
        return []

    decoded: list[DecodedValue] = []
    try:
        for result in zxingcpp.read_barcodes(image_array):
            if not result.text:
                continue
            decoded.append(
                DecodedValue(
                    value=result.text,
                    symbology=str(result.format),
                    bbox=_array_to_bbox(getattr(result, "position", None)),
                )
            )
    except Exception:
        return []
    return decoded


def _load_opencv_barcode_detector():
    if hasattr(cv2, "barcode_BarcodeDetector"):
        return cv2.barcode_BarcodeDetector()
    if hasattr(cv2, "barcode") and hasattr(cv2.barcode, "BarcodeDetector"):
        return cv2.barcode.BarcodeDetector()
    return None


def _decode_with_opencv_barcode(image_array: np.ndarray) -> list[DecodedValue]:
    detector = _load_opencv_barcode_detector()
    if detector is None:
        return []

    decoded: list[DecodedValue] = []
    try:
        success, decoded_info, decoded_types, points = detector.detectAndDecodeWithType(image_array)
    except Exception:
        return []

    if not success:
        return []

    if not isinstance(decoded_info, (list, tuple)):
        decoded_info = [decoded_info]
    if not isinstance(decoded_types, (list, tuple)):
        decoded_types = [decoded_types]
    point_groups = points if points is not None else []

    for raw_value, raw_type, point_set in zip(decoded_info, decoded_types, point_groups):
        if not raw_value:
            continue
        decoded.append(
            DecodedValue(
                value=str(raw_value),
                symbology=str(raw_type) if raw_type else "barcode",
                bbox=_array_to_bbox(point_set),
            )
        )
    return decoded


def _decode_with_pyzbar(image_array: np.ndarray) -> list[DecodedValue]:
    try:
        from pyzbar.pyzbar import decode as pyzbar_decode
    except Exception:
        return []

    decoded: list[DecodedValue] = []
    try:
        for item in pyzbar_decode(image_array):
            text = item.data.decode("utf-8", errors="ignore")
            if not text:
                continue
            rect = item.rect
            decoded.append(
                DecodedValue(
                    value=text,
                    symbology=str(item.type),
                    bbox=(rect.left, rect.top, rect.left + rect.width, rect.top + rect.height),
                )
            )
    except Exception:
        return []
    return decoded


def _decode_with_qr_detector(image_array: np.ndarray) -> list[DecodedValue]:
    detector = cv2.QRCodeDetector()
    if image_array.ndim == 2:
        source = image_array
    else:
        source = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    try:
        detected, decoded_info, points, _ = detector.detectAndDecodeMulti(source)
    except Exception:
        detected, decoded_info, points = False, [], None

    decoded: list[DecodedValue] = []
    if detected and points is not None:
        for raw_value, point_set in zip(decoded_info, points):
            if not raw_value:
                continue
            decoded.append(
                DecodedValue(
                    value=str(raw_value),
                    symbology="QR",
                    bbox=_array_to_bbox(point_set),
                )
            )
    return decoded


def _decode_with_wechat(image_array: np.ndarray) -> list[DecodedValue]:
    if not hasattr(cv2, "wechat_qrcode_WeChatQRCode"):
        return []

    model_dir = Path("wechat_qr_models")
    detector_proto = model_dir / "detect.prototxt"
    detector_model = model_dir / "detect.caffemodel"
    sr_proto = model_dir / "sr.prototxt"
    sr_model = model_dir / "sr.caffemodel"
    if not all(path.exists() for path in (detector_proto, detector_model, sr_proto, sr_model)):
        return []

    try:
        wechat = cv2.wechat_qrcode_WeChatQRCode(
            str(detector_proto),
            str(detector_model),
            str(sr_proto),
            str(sr_model),
        )
        decoded_strings, points = wechat.detectAndDecode(image_array)
    except Exception:
        return []

    decoded: list[DecodedValue] = []
    point_sets = points if points is not None else []
    for raw_value, point_set in zip(decoded_strings, point_sets):
        if not raw_value:
            continue
        decoded.append(
            DecodedValue(
                value=str(raw_value),
                symbology="WeChat-QR",
                bbox=_array_to_bbox(point_set),
            )
        )
    return decoded


def decoder_cascade(
    image_array: np.ndarray,
    preprocessing_variant: str,
    rotation: int,
    confidence: float | None,
    source_region_id: str | None,
) -> tuple[list[DecodedValue], list[str]]:
    trace: list[str] = []
    for decoder_name, decoder in (
        ("zxing-cpp", _decode_with_zxing),
        ("opencv-barcode", _decode_with_opencv_barcode),
        ("pyzbar", _decode_with_pyzbar),
        ("qr-detector", _decode_with_qr_detector),
        ("wechat-qr", _decode_with_wechat),
    ):
        decoded_values = decoder(image_array)
        trace.append(f"{decoder_name} on {preprocessing_variant} rotation {rotation}: {'success' if decoded_values else 'no result'}")
        if decoded_values:
            for decoded in decoded_values:
                decoded.decoder = decoder_name
                decoded.preprocessing = preprocessing_variant
                decoded.rotation = rotation
                decoded.confidence = confidence
                decoded.source_region_id = source_region_id
            return decoded_values, trace
    return [], trace


def explain_decode_failure(image: Image.Image, region_count: int) -> tuple[str, list[str]]:
    gray = np.array(image.convert("L"))
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = float(np.std(gray))

    causes: list[str] = []
    suggestions: list[str] = []

    if region_count == 0:
        causes.append("No barcode region was confidently detected.")
        suggestions.append("Hold the barcode closer and keep it centered in the frame.")
    else:
        causes.append("A barcode region was found, but the decode stage could not read the bars reliably.")

    if blur_score < 60:
        causes.append("The barcode appears blurred.")
        suggestions.append("Improve focus or hold the barcode more steadily.")
    if contrast < 35:
        causes.append("The crop has low contrast.")
        suggestions.append("Improve lighting and avoid glare on the label.")
    if gray.shape[1] < 180:
        causes.append("The barcode crop may be too small or too tight.")
        suggestions.append("Retry with the padded crop or capture the barcode closer.")
    suggestions.append("Keep the barcode as horizontal as possible.")
    suggestions.append("Make sure there is a white margin around the barcode.")

    message = " ".join(causes[:3]) or "No barcode could be decoded."
    unique_suggestions = []
    for suggestion in suggestions:
        if suggestion not in unique_suggestions:
            unique_suggestions.append(suggestion)
    return message, unique_suggestions[:4]


def _deduplicate_decoded_values(values: list[DecodedValue]) -> list[DecodedValue]:
    unique: dict[str, DecodedValue] = {}
    for decoded in values:
        key = normalize_code(decoded.value)
        if key and key not in unique:
            unique[key] = decoded
    return list(unique.values())


def decode_with_retry(
    image: Image.Image,
    confidence: float | None = None,
    source_region_id: str | None = None,
) -> tuple[list[DecodedValue], list[str]]:
    trace: list[str] = []
    for variant_name, variant_image in generate_preprocessing_variants(image):
        for rotation in (0, 90, 180, 270):
            rotated = _rotate_variant(variant_image, rotation)
            decoded_values, decoder_trace = decoder_cascade(
                image_array=rotated,
                preprocessing_variant=variant_name,
                rotation=rotation,
                confidence=confidence,
                source_region_id=source_region_id,
            )
            trace.extend(decoder_trace)
            if decoded_values:
                return _deduplicate_decoded_values(decoded_values), trace
    return [], trace


def _shift_bbox(bbox: tuple[int, int, int, int] | None, offset_x: int, offset_y: int) -> tuple[int, int, int, int] | None:
    if bbox is None:
        return None
    return (bbox[0] + offset_x, bbox[1] + offset_y, bbox[2] + offset_x, bbox[3] + offset_y)


def annotate_image(image: Image.Image, regions: list[DetectedRegion], decoded_values: list[DecodedValue]) -> Image.Image:
    annotated = image.convert("RGB").copy()
    canvas = ImageDraw.Draw(annotated)

    for region in regions:
        canvas.rectangle(region.bbox, outline="#0F766E", width=3)
        canvas.rectangle(region.padded_bbox, outline="#14B8A6", width=2)
        canvas.text(
            (region.padded_bbox[0], max(0, region.padded_bbox[1] - 18)),
            f"{region.crop_id} {region.confidence:.2f}",
            fill="#0F766E",
        )

    for decoded in decoded_values:
        if decoded.bbox is None:
            continue
        canvas.rectangle(decoded.bbox, outline="#DC2626", width=4)
        confidence_note = f" conf {decoded.confidence:.2f}" if decoded.confidence is not None else ""
        label = f"{decoded.symbology} · {decoded.decoder} · {decoded.preprocessing}"
        canvas.text(
            (decoded.bbox[0], max(0, decoded.bbox[1] - 18)),
            f"{label}{confidence_note}",
            fill="#DC2626",
        )

    return annotated


def mirror_scan_result_for_preview(scan_result: BarcodeScanResult, original_width: int) -> BarcodeScanResult:
    mirrored_regions: list[DetectedRegion] = []
    for region in scan_result.regions:
        bbox = (
            original_width - region.bbox[2],
            region.bbox[1],
            original_width - region.bbox[0],
            region.bbox[3],
        )
        padded_bbox = (
            original_width - region.padded_bbox[2],
            region.padded_bbox[1],
            original_width - region.padded_bbox[0],
            region.padded_bbox[3],
        )
        mirrored_regions.append(
            DetectedRegion(
                bbox=bbox,
                padded_bbox=padded_bbox,
                confidence=region.confidence,
                crop=region.crop,
                crop_id=region.crop_id,
            )
        )

    mirrored_values: list[DecodedValue] = []
    for decoded in scan_result.decoded_values:
        mirrored_bbox = None
        if decoded.bbox is not None:
            mirrored_bbox = (
                original_width - decoded.bbox[2],
                decoded.bbox[1],
                original_width - decoded.bbox[0],
                decoded.bbox[3],
            )
        mirrored_values.append(
            DecodedValue(
                value=decoded.value,
                symbology=decoded.symbology,
                bbox=mirrored_bbox,
                confidence=decoded.confidence,
                decoder=decoded.decoder,
                preprocessing=decoded.preprocessing,
                rotation=decoded.rotation,
                source_region_id=decoded.source_region_id,
            )
        )

    mirrored_base = scan_result.annotated_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    annotated_preview = annotate_image(mirrored_base, mirrored_regions, mirrored_values)
    return BarcodeScanResult(
        annotated_image=annotated_preview,
        regions=mirrored_regions,
        decoded_values=mirrored_values,
        decoder_trace=list(scan_result.decoder_trace),
        failure_message=scan_result.failure_message,
        suggestions=list(scan_result.suggestions),
        source_name=scan_result.source_name,
    )


def scan_barcode_image(image: Image.Image, source_name: str | None = None) -> BarcodeScanResult:
    rgb_image = image.convert("RGB")
    detected_regions = detect_regions(rgb_image)
    decoded_values: list[DecodedValue] = []
    decoder_trace: list[str] = []

    regions_to_scan = detected_regions or [
        DetectedRegion(
            bbox=(0, 0, rgb_image.width, rgb_image.height),
            padded_bbox=(0, 0, rgb_image.width, rgb_image.height),
            confidence=1.0,
            crop=rgb_image,
            crop_id="full-image",
        )
    ]

    for region in regions_to_scan:
        decoded, trace = decode_with_retry(region.crop, confidence=region.confidence, source_region_id=region.crop_id)
        decoder_trace.extend(trace)
        if decoded:
            for item in decoded:
                item.bbox = _shift_bbox(item.bbox, region.padded_bbox[0], region.padded_bbox[1]) or region.padded_bbox
                if item.confidence is None:
                    item.confidence = region.confidence
            decoded_values.extend(decoded)

    if not decoded_values and not detected_regions:
        decoded, trace = decode_with_retry(rgb_image, confidence=1.0, source_region_id="full-image")
        decoder_trace.extend(trace)
        for item in decoded:
            item.bbox = item.bbox or (0, 0, rgb_image.width, rgb_image.height)
        decoded_values.extend(decoded)

    decoded_values = _deduplicate_decoded_values(decoded_values)
    failure_message = None
    suggestions: list[str] = []
    if not decoded_values:
        failure_message, suggestions = explain_decode_failure(rgb_image, len(detected_regions))

    return BarcodeScanResult(
        annotated_image=annotate_image(rgb_image, detected_regions, decoded_values),
        regions=detected_regions,
        decoded_values=decoded_values,
        decoder_trace=decoder_trace,
        failure_message=failure_message,
        suggestions=suggestions,
        source_name=source_name,
    )


def retry_decode_crop(
    image: Image.Image,
    confidence: float | None = None,
    source_region_id: str | None = None,
    source_name: str | None = None,
) -> BarcodeScanResult:
    rgb_image = image.convert("RGB")
    decoded_values, decoder_trace = decode_with_retry(
        rgb_image,
        confidence=confidence,
        source_region_id=source_region_id,
    )
    for item in decoded_values:
        item.bbox = item.bbox or (0, 0, rgb_image.width, rgb_image.height)
        if item.confidence is None:
            item.confidence = confidence

    failure_message = None
    suggestions: list[str] = []
    if not decoded_values:
        failure_message, suggestions = explain_decode_failure(rgb_image, 1)

    detector_confidence = float(confidence) if confidence is not None else 0.0
    retry_region = DetectedRegion(
        bbox=(0, 0, rgb_image.width, rgb_image.height),
        padded_bbox=(0, 0, rgb_image.width, rgb_image.height),
        confidence=detector_confidence,
        crop=rgb_image,
        crop_id=source_region_id or "selected-crop",
    )
    return BarcodeScanResult(
        annotated_image=annotate_image(rgb_image, [retry_region], decoded_values),
        regions=[retry_region],
        decoded_values=_deduplicate_decoded_values(decoded_values),
        decoder_trace=decoder_trace,
        failure_message=failure_message,
        suggestions=suggestions,
        source_name=source_name,
    )


def decode_image(image: Image.Image) -> list[DecodedValue]:
    return retry_decode_crop(image).decoded_values


def load_order_dataframe(uploaded_file) -> pd.DataFrame:
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        dataframe = pd.read_csv(uploaded_file, dtype=str, keep_default_na=False).fillna("")
    else:
        dataframe = pd.read_excel(uploaded_file, dtype=str, engine="openpyxl").fillna("")
    dataframe.columns = [normalize_lookup_text(column) for column in dataframe.columns]
    for column in dataframe.columns:
        dataframe[column] = dataframe[column].map(normalize_lookup_text)
    return dataframe


def guess_lookup_columns(dataframe: pd.DataFrame) -> list[str]:
    preferred_keywords = [
        "barcode",
        "sku",
        "productcode",
        "product_code",
        "productid",
        "itemcode",
        "item_code",
        "orderid",
        "order_id",
        "tracking",
        "code",
    ]

    ranked_columns: list[tuple[int, str]] = []
    for column in dataframe.columns:
        normalized = normalize_code(column)
        rank = next((index for index, keyword in enumerate(preferred_keywords) if keyword.upper().replace("_", "") in normalized), None)
        if rank is not None:
            ranked_columns.append((rank, column))

    if ranked_columns:
        return [column for _, column in sorted(ranked_columns)]
    return list(dataframe.columns[:3])


def lookup_orders_detailed(
    dataframe: pd.DataFrame,
    decoded_code: str,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    if dataframe.empty or not decoded_code:
        return dataframe.iloc[0:0], columns or [], []

    selected_columns = columns or guess_lookup_columns(dataframe)
    normalized_target = normalize_code(decoded_code)
    if not normalized_target:
        return dataframe.iloc[0:0], selected_columns, []

    match_mask = pd.Series(False, index=dataframe.index)
    matched_columns: list[str] = []
    for column in selected_columns:
        normalized_series = dataframe[column].fillna("").map(normalize_code)
        column_match = normalized_series.eq(normalized_target) | normalized_series.str.contains(normalized_target, regex=False)
        if bool(column_match.any()):
            matched_columns.append(column)
        match_mask = match_mask | column_match

    matched_rows = dataframe[match_mask].copy()
    return matched_rows, selected_columns, matched_columns


def lookup_orders(dataframe: pd.DataFrame, decoded_code: str, columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    matched_rows, selected_columns, _ = lookup_orders_detailed(dataframe, decoded_code, columns)
    return matched_rows, selected_columns


def build_barcode_log_record(
    decoded_code: str,
    matched_rows: pd.DataFrame,
    searched_columns: list[str],
    source: str,
    scan_result: BarcodeScanResult | None = None,
    matched_columns: list[str] | None = None,
) -> dict[str, Any]:
    preview_rows: list[dict[str, Any]] = []
    if not matched_rows.empty:
        for row in matched_rows.head(5).to_dict(orient="records"):
            preview_rows.append({key: str(value) for key, value in row.items()})

    primary = scan_result.primary_decoded if scan_result else None
    top_confidence = scan_result.regions[0].confidence if scan_result and scan_result.regions else None
    first_crop_id = scan_result.regions[0].crop_id if scan_result and scan_result.regions else None
    return {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_image_name": scan_result.source_name if scan_result else None,
        "decoded_code": decoded_code,
        "decode_success": bool(primary),
        "decoded_type": primary.symbology if primary else None,
        "decoder_used": primary.decoder if primary else None,
        "preprocessing_variant": primary.preprocessing if primary else None,
        "rotation": primary.rotation if primary else None,
        "detector_confidence": top_confidence,
        "crop_id": first_crop_id,
        "matched": not matched_rows.empty,
        "matched_rows_count": int(len(matched_rows)),
        "matched_columns": matched_columns or [],
        "searched_columns": searched_columns,
        "source": source,
        "failure_message": scan_result.failure_message if scan_result else None,
        "suggestions": scan_result.suggestions if scan_result else [],
        "decoder_trace": scan_result.decoder_trace[:30] if scan_result else [],
        "preview_rows": preview_rows,
    }


def save_barcode_log(log_record: dict[str, Any], log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = log_record.get("timestamp", datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")).replace(":", "").replace("-", "")
    decoded = normalize_code(log_record.get("decoded_code", "")) or "barcode"
    output_path = log_dir / f"{timestamp}_{decoded}.json"
    output_path.write_text(json.dumps(log_record, ensure_ascii=True, indent=2), encoding="utf-8")
    return output_path


def load_barcode_logs(log_dir: Path) -> list[dict[str, Any]]:
    if not log_dir.exists():
        return []
    records: list[dict[str, Any]] = []
    for log_file in sorted(log_dir.glob("*.json"), reverse=True):
        try:
            payload = json.loads(log_file.read_text(encoding="utf-8"))
            payload["log_file"] = str(log_file)
            records.append(payload)
        except Exception:
            continue
    return records


class LiveBarcodeProcessor:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frame_count = 0
        self.latest_result: BarcodeScanResult | None = None
        self.latest_code: str | None = None

    def recv(self, frame):
        import av

        original_bgr = frame.to_ndarray(format="bgr24")
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        preview_bgr = cv2.flip(original_bgr, 1)
        self.frame_count += 1

        if self.frame_count % 8 == 0:
            pil_image = Image.fromarray(original_rgb)
            scan_result = scan_barcode_image(pil_image, source_name="live-camera")
            preview_result = mirror_scan_result_for_preview(scan_result, pil_image.width)
            with self.lock:
                self.latest_result = preview_result
                self.latest_code = scan_result.primary_decoded.value if scan_result.primary_decoded else None
            preview_bgr = cv2.cvtColor(np.array(preview_result.annotated_image), cv2.COLOR_RGB2BGR)
        else:
            with self.lock:
                if self.latest_result is not None:
                    preview_bgr = cv2.cvtColor(np.array(self.latest_result.annotated_image), cv2.COLOR_RGB2BGR)

        return av.VideoFrame.from_ndarray(preview_bgr, format="bgr24")

    def snapshot(self) -> tuple[BarcodeScanResult | None, str | None]:
        with self.lock:
            return self.latest_result, self.latest_code
