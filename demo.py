import multiprocessing as mp
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from PointRend.point_rend import SemSegDatasetMapper, add_pointrend_config

WINDOW_NAME = "Traffic Seg"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        # default="/home/sk49/new_workspace/trj/detectron2-master/projects/PointRend/configs/SemanticSegmentation/pointrend_semantic_R_50_FPN_1x_coco.yaml",
        default="PointRend/configs/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes.yaml",
        # default="/cache/user-job-dir/demo/PointRend/configs/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument(
        "--video-input",
        # default='/home/sk49/new_workspace/trj/CenterTrack/videos/2mins.mp4',
        # default="/cache/user-job-dir/demo/2mins.mp4",
        help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        # default=['/home/sk49/new_workspace/trj/img/test.png'],
        # default=['/cache/user-job-dir/demo/img/test.png'],
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        # default='/home/sk49/new_workspace/trj',
        # default='/cache/user-job-dir/demo',
        default='output/',
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS', "models/pointrend_sem_city_1x.pkl"],
        # default=['MODEL.WEIGHTS', "/cache/user-job-dir/demo/model_final_5f3665.pkl"],
        nargs=argparse.REMAINDER,
    )
    # parser.add_argument(
    #     "--tracks",
    #     nargs="+",
    #     # default=['/home/sk49/new_workspace/trj/img/test.png'],
    #     # default=['/cache/user-job-dir/deep_sort_pytorch/output'],
    # )
    # parser.add_argument(
    #     "--data_url",
    #     help='huaweiyun',
    # )
    # parser.add_argument(
    #     "--init_method",
    #     help='huaweiyun',
    # )
    # parser.add_argument(
    #     "--train_url",
    #     help='huaweiyun',
    # )
    return parser



if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    basename = os.path.basename(args.video_input)
    video_input_name = basename.split('/')[-1:][0]
    demo = VisualizationDemo(cfg,"input/tracks_"+video_input_name.split('.')[0]+".txt")

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, 'test2.png')
                    # out_filename = 'test2.png'
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                # out_filename = os.path.join(args.output, 'test2.png')
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))


        if args.output:
            if os.path.isdir(args.output):
                # output_fname = os.path.join(args.output, basename)
                # output_fname = os.path.splitext(output_fname)[0] + ".mkv"
                # output_fname = '2min_result_coco.avi'
                output_fname_first=args.output+'/res-'+video_input_name.split('.')[0]
                output_fname_last=video_input_name.split('.')[-1:][0]
                output_fname = output_fname_first+'.'+output_fname_last
                while os.path.isfile(output_fname):
                    output_fname = output_fname.split('.')[0]+'-new.'+output_fname_last
                # print(output_fname)
            else:
                output_fname = args.output
            # assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                # fourcc=cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                # fourcc=cv2.VideoWriter_fourcc(*'MJPG'),
                fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        with open(output_fname.split('.')[0]+'.txt', "w") as fout:
            from predictor import all_cnt,categories,res_dect
            for i in range(30):
                cid=str(i)
                fout.write(categories[cid] + " : " + str(all_cnt[cid]) + '\n')
                # inslist = [x["insid"] for x in res_dect["category_" + cid]["instances"].values()]
                # seglist = [str(x/frames_per_second+1)+'s' for x in range(num_frames) if x%300==0]
                # if (len(inslist) > 0):
                #     fout.write(
                #         categories[cid] + ": " + str(track_cnt[cid]) + ", track instances id: " + str(inslist).replace(
                #             "'", "").replace("[", "").replace(']', "") + '\n')
                # elif (track_cnt[cid] > 0):
                #     fout.write(
                #         categories[cid] + ": " + str(track_cnt[cid]) + ", segment object time: " + str(seglist).replace(
                #             "'", "").replace("[", "").replace(']', "") + '\n')
                # else:
                #     fout.write(categories[cid] + ": " + str(track_cnt[cid]) + '\n')
        fout.close()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
