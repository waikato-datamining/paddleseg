from datetime import datetime
import numpy as np
import traceback
import cv2

from rdh import Container, MessageContainer, create_parser, configure_redis, run_harness, log
from predict_common import prediction_to_data, PREDICTION_FORMATS, PREDICTION_FORMAT_GRAYSCALE, load_model, classes_dict
import paddle
from paddleseg.core.predict import preprocess
from paddleseg.core import infer


def process_image(msg_cont):
    """
    Processes the message container, loading the image from the message and forwarding the predictions.

    :param msg_cont: the message container to process
    :type msg_cont: MessageContainer
    """
    config = msg_cont.params.config

    try:
        start_time = datetime.now()

        array = np.frombuffer(msg_cont.message['data'], np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)

        with paddle.no_grad():
            # TODO binary data as input?
            data = preprocess(image, transforms)
            pred, _ = infer.inference(
                model,
                data['img'],
                trans_info=data['trans_info'],
                is_slide=config.is_slide,
                stride=config.stride,
                crop_size=config.crop_size,
                use_multilabel=False)
        out_data = prediction_to_data(pred, config.prediction_format,
                                      mask_nth=config.mask_nth, classes=config.classes)
        msg_cont.params.redis.publish(msg_cont.params.channel_out, out_data)

        if config.verbose:
            log("process_images - prediction image published: %s" % msg_cont.params.channel_out)
            end_time = datetime.now()
            processing_time = end_time - start_time
            processing_time = int(processing_time.total_seconds() * 1000)
            log("process_images - finished processing image: %d ms" % processing_time)

    except KeyboardInterrupt:
        msg_cont.params.stopped = True
    except:
        log("process_images - failed to process: %s" % traceback.format_exc())


if __name__ == '__main__':
    parser = create_parser('PaddleSeg - Prediction (Redis)', prog="paddleseg_predict_redis", prefix="redis_")
    parser.add_argument('--config', help='Path to the config file', required=True, default=None)
    parser.add_argument('--model_path', help='Path to the trained model (.pdparams file)', required=True, default=None)
    parser.add_argument('--device', help='The device to use', default="gpu:0")
    parser.add_argument('--prediction_format', default=PREDICTION_FORMAT_GRAYSCALE, choices=PREDICTION_FORMATS, help='The format for the prediction images')
    parser.add_argument('--labels', help='Path to the text file with the labels; one per line, including background', required=True, default=None)
    parser.add_argument('--is_slide', help='Whether to predict images in sliding window method', action='store_true')
    parser.add_argument('--crop_size', nargs=2, help='The crop size of sliding window, the first is width and the second is height. For example, `--crop_size 512 512`', type=int)
    parser.add_argument('--stride', nargs=2, help='The stride of sliding window, the first is width and the second is height. For example, `--stride 512 512`', type=int)
    parser.add_argument('--mask_nth', type=int, help='To speed polygon detection up, use every nth row and column only (OPEX format only)', required=False, default=1)
    parser.add_argument('--verbose', action='store_true', help='Whether to output more logging info', required=False, default=False)
    parsed = parser.parse_args()

    try:
        model, transforms = load_model(parsed.config, parsed.model_path, parsed.device)

        config = Container()
        config.model = model
        config.transforms = transforms
        config.prediction_format = parsed.prediction_format
        config.is_slide = parsed.is_slide
        config.crop_size = parsed.crop_size
        config.stride = parsed.stride
        config.mask_nth = parsed.mask_nth
        config.classes = classes_dict(parsed.labels)
        config.verbose = parsed.verbose

        params = configure_redis(parsed, config=config)
        run_harness(params, process_image)

    except Exception as e:
        print(traceback.format_exc())
