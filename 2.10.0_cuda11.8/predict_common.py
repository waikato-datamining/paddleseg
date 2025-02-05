import cv2
import io
import numpy as np
import os

from datetime import datetime
from typing import Union, Dict, Tuple
from opex import ObjectPredictions, ObjectPrediction, BBox, Polygon
from PIL import Image
from smu import mask_to_polygon, polygon_to_lists
from simple_palette_utils import default_palette
from paddleseg.cvlibs import Config, SegBuilder
from paddleseg.transforms import Compose
import paddle
import paddleseg.utils as utils

PREDICTION_FORMAT_GRAYSCALE = "grayscale"
PREDICTION_FORMAT_BLUECHANNEL = "bluechannel"
PREDICTION_FORMAT_INDEXED = "indexed"
PREDICTION_FORMAT_BMP = "bmp"
PREDICTION_FORMAT_OPEX = "opex"
PREDICTION_FORMATS = [
    PREDICTION_FORMAT_GRAYSCALE,
    PREDICTION_FORMAT_BLUECHANNEL,
    PREDICTION_FORMAT_INDEXED,
    PREDICTION_FORMAT_BMP,
    PREDICTION_FORMAT_OPEX,
]


def load_model(config: str, model_path: str, device: str) -> Tuple:
    """
    Loads the model.

    :param config: the path to the config file
    :type config: str
    :param model_path: the path to the trained model (.pdparams file)
    :type model_path: str
    :param device: the device to use, e.g., cuda:0 or cpu
    :type device: str
    :return: the tuple of model and validation transformations
    :rtype: tuple
    """
    utils.set_device(device)
    cfg = Config(config)
    builder = SegBuilder(cfg)
    model = builder.model
    utils.utils.load_entire_model(model, model_path)
    model.eval()
    transforms = Compose(builder.val_transforms)
    return model, transforms


def mask_to_opex(pr_mask, id_: str, ts: str, mask_nth: int = 1, classes: Dict[int,str] = None) -> ObjectPredictions:
    """
    Turns the segmentation mask into OPEX predictions.

    :param pr_mask: the mask to convert
    :param id_: the ID to use for the predictions
    :type id_: str
    :param ts: the timestamp to use
    :type ts: str
    :param mask_nth: the contour tracing can be slow for large masks, by using only every nth row/col, this can be sped up dramatically
    :type mask_nth: int
    :param classes: the index/label relationship dictionary
    :type classes: dict
    :return: the opex predictions
    :rtype: ObjectPredictions
    """
    pr_mask = np.squeeze(pr_mask)
    values = np.unique(pr_mask)
    pred_objs = []
    for value in values:
        if value == 0:
            continue
        sub_mask = np.where(pr_mask == value, pr_mask, 0)
        polys = mask_to_polygon(sub_mask, mask_nth=mask_nth)
        for poly in polys:
            px, py = polygon_to_lists(poly, swap_x_y=True, normalize=False, as_type="int")
            x0 = min(px)
            y0 = min(py)
            x1 = max(px)
            y1 = max(py)
            if (x0 < x1) and (y0 < y1):
                bbox = BBox(left=x0, top=y0, right=x1, bottom=y1)
                points = []
                for x, y in zip(px, py):
                    points.append((x, y))
                poly = Polygon(points=points)
                label = "object"
                if (classes is not None) and (value in classes):
                    label = classes[value]
                opex_obj = ObjectPrediction(label=label, bbox=bbox, polygon=poly)
                pred_objs.append(opex_obj)
    return ObjectPredictions(id=id_, timestamp=ts, objects=pred_objs)


def prediction_to_file(prediction, prediction_format: str, path: str, mask_nth: int = 1, classes: Dict[int,str] = None) -> str:
    """
    Saves the mask prediction to disk as image using the specified image format.

    :param prediction: the paddleseg prediction object
    :param prediction_format: the image format to use
    :type prediction_format: str
    :param path: the path to save the image to
    :type path: str
    :param mask_nth: the contour tracing can be slow for large masks, by using only every nth row/col, this can be sped up dramatically
    :type mask_nth: int
    :param classes: the index/label relationship dictionary
    :type classes: dict
    :return: the filename the predictions were saved under
    :rtype: str
    """
    if prediction_format not in PREDICTION_FORMATS:
        raise Exception("Unsupported format: %s" % prediction_format)

    pr_mask = paddle.squeeze(prediction)
    pr_mask = pr_mask.numpy().astype('uint8')

    if prediction_format == PREDICTION_FORMAT_GRAYSCALE:
        cv2.imwrite(path, pr_mask)
    elif prediction_format == PREDICTION_FORMAT_BLUECHANNEL:
        pr_mask = cv2.cvtColor(pr_mask, cv2.COLOR_GRAY2RGB)
        pr_mask[:, :, 1] = np.zeros([pr_mask.shape[0], pr_mask.shape[1]])
        pr_mask[:, :, 2] = np.zeros([pr_mask.shape[0], pr_mask.shape[1]])
        cv2.imwrite(path, pr_mask)
    elif prediction_format == PREDICTION_FORMAT_INDEXED:
        pr_mask = np.squeeze(pr_mask)
        pr_pil = Image.fromarray(pr_mask, "P")
        pr_pil.putpalette(default_palette())
        pr_pil.save(path)
    elif prediction_format == PREDICTION_FORMAT_BMP:
        path = os.path.splitext(path)[0] + ".bmp"
        pr_mask = np.squeeze(pr_mask)
        pr_pil = Image.fromarray(pr_mask, "P")
        pr_pil.putpalette(default_palette())
        pr_pil.save(path)
    elif prediction_format == PREDICTION_FORMAT_OPEX:
        path = os.path.splitext(path)[0] + ".json"
        opex_preds = mask_to_opex(pr_mask, os.path.basename(path), str(datetime.now()), mask_nth=mask_nth, classes=classes)
        opex_preds.save_json_to_file(path)
    else:
        raise Exception("Unhandled format: %s" % prediction_format)

    return path


def prediction_to_data(prediction, prediction_format: str, mask_nth: int = 1, classes: Dict[int,str] = None) -> Union[bytes, str]:
    """
    Turns the mask prediction into bytes using the specified image format.

    :param prediction: the paddleseg prediction object
    :param prediction_format: the image format to use
    :type prediction_format: str
    :param mask_nth: the contour tracing can be slow for large masks, by using only every nth row/col, this can be sped up dramatically
    :type mask_nth: int
    :param classes: the index/label relationship dictionary
    :type classes: dict
    :return: the generated image
    :rtype: bytes
    """
    if prediction_format not in PREDICTION_FORMATS:
        raise Exception("Unsupported format: %s" % prediction_format)

    pr_mask = paddle.squeeze(prediction)
    pr_mask = pr_mask.numpy().astype('uint8')

    if prediction_format == PREDICTION_FORMAT_GRAYSCALE:
        result = cv2.imencode('.png', pr_mask)[1].tobytes()
    elif prediction_format == PREDICTION_FORMAT_BLUECHANNEL:
        pr_mask = cv2.cvtColor(pr_mask, cv2.COLOR_GRAY2RGB)
        pr_mask[:, :, 1] = np.zeros([pr_mask.shape[0], pr_mask.shape[1]])
        pr_mask[:, :, 2] = np.zeros([pr_mask.shape[0], pr_mask.shape[1]])
        result = cv2.imencode('.png', pr_mask)[1].tobytes()
    elif prediction_format == PREDICTION_FORMAT_INDEXED:
        pr_mask = np.squeeze(pr_mask)
        pr_pil = Image.fromarray(pr_mask, "P")
        pr_pil.putpalette(default_palette())
        buffer = io.BytesIO()
        pr_pil.save(buffer, format="PNG")
        result = buffer.getvalue()
    elif prediction_format == PREDICTION_FORMAT_BMP:
        pr_mask = np.squeeze(pr_mask)
        pr_pil = Image.fromarray(pr_mask, "P")
        pr_pil.putpalette(default_palette())
        buffer = io.BytesIO()
        pr_pil.save(buffer, format="BMP")
        result = buffer.getvalue()
    elif prediction_format == PREDICTION_FORMAT_OPEX:
        ts = str(datetime.now())
        opex_preds = mask_to_opex(pr_mask, ts, ts, mask_nth=mask_nth, classes=classes)
        result = opex_preds.to_json_string()
    else:
        raise Exception("Unhandled format: %s" % prediction_format)

    return result
