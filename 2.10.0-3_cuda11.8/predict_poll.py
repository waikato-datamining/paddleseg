import os
import argparse
from image_complete import auto
import traceback

from sfp import Poller
from predict_common import prediction_to_file, PREDICTION_FORMATS, PREDICTION_FORMAT_GRAYSCALE, load_model, classes_dict
import paddle
from paddleseg.core.predict import preprocess
from paddleseg.core import infer


SUPPORTED_EXTS = [".jpg", ".jpeg"]
""" supported file extensions (lower case). """


def check_image(fname, poller):
    """
    Check method that ensures the image is valid.

    :param fname: the file to check
    :type fname: str
    :param poller: the Poller instance that called the method
    :type poller: Poller
    :return: True if complete
    :rtype: bool
    """
    result = auto.is_image_complete(fname)
    poller.debug("Image complete:", fname, "->", result)
    return result


def process_image(fname, output_dir, poller):
    """
    Method for processing an image.

    :param fname: the image to process
    :type fname: str
    :param output_dir: the directory to write the image to
    :type output_dir: str
    :param poller: the Poller instance that called the method
    :type poller: Poller
    :return: the list of generated output files
    :rtype: list
    """
    result = []

    try:
        # TODO batches?
        with paddle.no_grad():
            data = preprocess(fname, transforms)
            pred, _ = infer.inference(
                model,
                data['img'],
                trans_info=data['trans_info'],
                is_slide=poller.params.is_slide,
                stride=poller.params.stride,
                crop_size=poller.params.crop_size,
                use_multilabel=False)
        fname_out = os.path.join(output_dir, os.path.splitext(os.path.basename(fname))[0] + ".png")
        fname_out = prediction_to_file(pred, poller.params.prediction_format, fname_out,
                                       mask_nth=poller.params.mask_nth, classes=poller.params.classes)
        result.append(fname_out)
    except KeyboardInterrupt:
        poller.keyboard_interrupt()
    except:
        poller.error("Failed to process image: %s\n%s" % (fname, traceback.format_exc()))
    return result


def predict_on_images(input_dir, model, transforms, output_dir, tmp_dir, prediction_format="grayscale", labels=None,
                      is_slide=False, crop_size=None, stride=False, mask_nth=1, poll_wait=1.0, continuous=False, use_watchdog=False, watchdog_check_interval=10.0,
                      delete_input=False, verbose=False, quiet=False):
    """
    Method for performing predictions on images.

    :param input_dir: the directory with the images
    :type input_dir: str
    :param model: the paddleseg trained model
    :param transforms: the transformations to apply to the images
    :param output_dir: the output directory to move the images to and store the predictions
    :type output_dir: str
    :param tmp_dir: the temporary directory to store the predictions until finished, use None if not to use
    :type tmp_dir: str
    :param prediction_format: the format to use for the prediction images (grayscale/bluechannel)
    :type prediction_format: str
    :param labels: the path to the file with the labels (one per line, including background)
    :type labels: str
    :param is_slide: Whether to predict images in sliding window method
    :type is_slide: bool
    :param crop_size: The crop size of sliding window, the first is width and the second is height.
    :type crop_size: tuple
    :param stride: The stride of sliding window, the first is width and the second is height.
    :type stride: bool
    :param mask_nth: the contour tracing can be slow for large masks, by using only every nth row/col, this can be sped up dramatically
    :type mask_nth: int
    :param poll_wait: the amount of seconds between polls when not in watchdog mode
    :type poll_wait: float
    :param continuous: whether to poll continuously
    :type continuous: bool
    :param use_watchdog: whether to react to file creation events rather than use fixed-interval polling
    :type use_watchdog: bool
    :param watchdog_check_interval: the interval for the watchdog process to check for files that were missed due to potential race conditions
    :type watchdog_check_interval: float
    :param delete_input: whether to delete the input images rather than moving them to the output directory
    :type delete_input: bool
    :param verbose: whether to output more logging information
    :type verbose: bool
    :param quiet: whether to suppress output
    :type quiet: bool
    """

    poller = Poller()
    poller.input_dir = input_dir
    poller.output_dir = output_dir
    poller.tmp_dir = tmp_dir
    poller.extensions = SUPPORTED_EXTS
    poller.delete_input = delete_input
    poller.progress = not quiet
    poller.verbose = verbose
    poller.check_file = check_image
    poller.process_file = process_image
    poller.poll_wait = poll_wait
    poller.continuous = continuous
    poller.use_watchdog = use_watchdog
    poller.watchdog_check_interval = watchdog_check_interval
    poller.params.model = model
    poller.params.transforms = transforms
    poller.params.prediction_format = prediction_format
    poller.params.is_slide = is_slide
    poller.params.crop_size = crop_size
    poller.params.stride = stride
    poller.params.mask_nth = mask_nth
    poller.params.classes = classes_dict(labels)
    poller.poll()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PaddleSeg - Prediction", prog="paddleseg_predict_poll", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', help='Path to the config file', required=True, default=None)
    parser.add_argument('--model_path', help='Path to the trained model (.pdparams file)', required=True, default=None)
    parser.add_argument('--device', help='The device to use', default="gpu:0")
    parser.add_argument('--labels', help='Path to the text file with the labels; one per line, including background', required=True, default=None)
    parser.add_argument('--prediction_in', help='Path to the test images', required=True, default=None)
    parser.add_argument('--prediction_out', help='Path to the output csv files folder', required=True, default=None)
    parser.add_argument('--prediction_tmp', help='Path to the temporary csv files folder', required=False, default=None)
    parser.add_argument('--prediction_format', default=PREDICTION_FORMAT_GRAYSCALE, choices=PREDICTION_FORMATS, help='The format for the prediction images')
    parser.add_argument('--is_slide', help='Whether to predict images in sliding window method', action='store_true')
    parser.add_argument('--crop_size', nargs=2, help='The crop size of sliding window, the first is width and the second is height. For example, `--crop_size 512 512`', type=int)
    parser.add_argument('--stride', nargs=2, help='The stride of sliding window, the first is width and the second is height. For example, `--stride 512 512`', type=int)
    parser.add_argument('--mask_nth', type=int, help='To speed polygon detection up, use every nth row and column only (OPEX format only)', required=False, default=1)
    parser.add_argument('--poll_wait', type=float, help='poll interval in seconds when not using watchdog mode', required=False, default=1.0)
    parser.add_argument('--continuous', action='store_true', help='Whether to continuously load test images and perform prediction', required=False, default=False)
    parser.add_argument('--use_watchdog', action='store_true', help='Whether to react to file creation events rather than performing fixed-interval polling', required=False, default=False)
    parser.add_argument('--watchdog_check_interval', type=float, help='check interval in seconds for the watchdog', required=False, default=10.0)
    parser.add_argument('--delete_input', action='store_true', help='Whether to delete the input images rather than move them to --prediction_out directory', required=False, default=False)
    parser.add_argument('--verbose', action='store_true', help='Whether to output more logging info', required=False, default=False)
    parser.add_argument('--quiet', action='store_true', help='Whether to suppress output', required=False, default=False)
    parsed = parser.parse_args()

    try:
        model, transforms = load_model(parsed.config, parsed.model_path, parsed.device)

        # Performing the prediction and producing the predictions files
        predict_on_images(parsed.prediction_in, model, transforms, parsed.prediction_out, parsed.prediction_tmp,
                          prediction_format=parsed.prediction_format, labels=parsed.labels, mask_nth=parsed.mask_nth,
                          is_slide=parsed.is_slide, stride=parsed.stride, crop_size=parsed.crop_size, continuous=parsed.continuous,
                          use_watchdog=parsed.use_watchdog, watchdog_check_interval=parsed.watchdog_check_interval,
                          delete_input=parsed.delete_input, verbose=parsed.verbose, quiet=parsed.quiet)

    except Exception as e:
        print(traceback.format_exc())
