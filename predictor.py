# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch
import numpy as np

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer,VisImage,colormap

categories={"0":"Road", "1":"Sidewalk", "2":"Building", "3":"Wall", "4":"Fence",
            "5":"Pole", "6":"traffic light", "7":"traffic sign", "8":"Vegetation",
            "9":"Terrain", "10":"Sky", "11":"Person", "12":"Rider",
            "13":"Car", "14":"Truck", "15":"Bus", "16":"Train",
            "17":"Motorcycle", "18":"Bicycle", "19":"Others", "20":"Airplane",
            "21":"Cat", "22":"Dog", "23":"Umbrella", "24":"Handbag",
            "25":"Suitcase", "26":"Cellphone", "27":"Stopsign", "28":"Parking_meter",
            "29":"Bench"}

track_cnt = { "0": 0,"1": 0, "2": 0, "3": 0, "4": 0, "5": 0,
                    "6": 0, "7": 0, "8": 0, "9": 0, "10": 0,
                    "11": 0, "12": 0, "13": 0, "14": 0, "15": 0,
                    "16": 0, "17": 0, "18": 0, "19": 0, "20":0,
                    "21": 0, "22": 0, "23": 0, "24": 0, "25": 0,
                    "26": 0, "27": 0, "28": 0, "29":0
                    }
all_cnt = { "0": 0,"1": 0, "2": 0, "3": 0, "4": 0, "5": 0,
                    "6": 0, "7": 0, "8": 0, "9": 0, "10": 0,
                    "11": 0, "12": 0, "13": 0, "14": 0, "15": 0,
                    "16": 0, "17": 0, "18": 0, "19": 0, "20":0,
                    "21": 0, "22": 0, "23": 0, "24": 0, "25": 0,
                    "26": 0, "27": 0, "28": 0, "29":0
                    }

seg_cnt = { "0": 0,"1": 0, "2": 0, "3": 0, "4": 0, "5": 0,
                    "6": 0, "7": 0, "8": 0, "9": 0, "10": 0,
                    "11": 0, "12": 0, "13": 0, "14": 0, "15": 0,
                    "16": 0, "17": 0, "18": 0, "19": 0,"20":0,
                    "21": 0, "22": 0, "23": 0, "24": 0, "25": 0,
                    "26": 0, "27": 0, "28": 0, "29": 0}

res_dect = {}
tracks={}
def motcatcnt(category_mot):
    map={"0":"11", "1":"18", "2":"13", "3":"17", "4":"20", "5":"15",
         "7":"14", "9":"6", "15":"21", "16":"22", "25":"23", "26":"24",
         "28":"25", "67":"26", "11":"27", "12":"28","13":"29"}
    return map[category_mot]

def statis_detect(tracks_path):
    # 定义数据结构，并初始化
    #用于统计结果
    # res_dect = {}
    for i in range(30):
        id_category = str(i)
        res_dect["category_" + id_category] = {"category": categories[id_category], "cnt_ins": 0, "instances": {}}
    #用于画框
    # tracks={}

    with open(tracks_path, "r") as fin:
        oreslist=fin.readlines()
        oreslist = [x.split(',') for x in oreslist if len(x) >=7]
        for ores in oreslist:
            frameid=ores[0]     #帧id
            insid=ores[1]       #实例id
            id_category=motcatcnt(ores[2])    #分类，类别id,映射
            bbox=[int(ores[3]),int(ores[4]),int(ores[5]),int(ores[6])]  #位置信息

            # 统计：每一个类别出现多少个实例(每个category中不重复的instance_id)，每个实例出现在的帧数，以及在该帧中的位置（x,y,w,h）
            # #给类别添加实例
            # 找不到以该insid为主键的instances对象，该实例没有加入类别中，需要添加实例
            if(res_dect["category_" + id_category]["instances"].get("ins_"+str(insid)) == None):
                cateid = res_dect["category_" + id_category]["cnt_ins"] + 1  # 在类别中的id是当前类别已有数目+1
                res_dect["category_" + id_category]["instances"]["ins_" + str(insid)] = {"insid": insid,
                                                                                          "cateid": cateid,
                                                                                          "frames": {}}  # 添加实例
                res_dect["category_" + id_category]["cnt_ins"] = cateid  # 更新当前类别已有数目
                # print(res_dect)
            # #给实例添加帧和位置信息
            res_dect["category_" + id_category]["instances"]["ins_" + str(insid)]["frames"]["frame_" + frameid] = {
                "frameid": frameid, "bbox": bbox}

            # 统计：每一帧出现的每一个实例：instance_id(他是从视频开始出现的第几个实例) category(他的类别) id_category(他是该类别中的第几个出现) 他的位置
            # 找不到以该frameid为主键的tracks对象，该帧没有创建：
            if (tracks.get(frameid) == None):
                tracks[frameid]=[]
            cateid = res_dect["category_" + id_category]["instances"]["ins_" + str(insid)]["cateid"]
            tracks[frameid].append({"category_id": id_category, "cateid":cateid, "insid": insid,"bbox": bbox})

        for i in range(30):
            all_cnt[str(i)]=res_dect["category_"+str(i)]["cnt_ins"]

        return {"res_dect":res_dect,"tracks": tracks}

class VisualizationDemo(object):
    def __init__(self, cfg, tracks_path,instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        # print(tracks_path)
        self.tracks = statis_detect(tracks_path)['tracks']
        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        path = 'tmp.txt'  # 文件路径
        f = open(path, 'w', encoding='utf-8')
        # print(predictions['sem_seg'].shape)
        for k, v in predictions.items():
            s1 = v.argmax(dim=0)
            s2 = str(s1)  # 把字典的值转换成字符型
            # f.write(k + ':')
            f.write(s2 + '\n')
        f.close()
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions, id_frame):
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # vis_frame = VisImage(frame)
            vis_frame = frame
            # if "panoptic_seg" in predictions:
            #     panoptic_seg, segments_info = predictions["panoptic_seg"]
            #     vis_frame = video_visualizer.draw_panoptic_seg_predictions(
            #         frame, panoptic_seg.to(self.cpu_device), segments_info
            #     )
            # elif "instances" in predictions:
            #     predictions = predictions["instances"].to(self.cpu_device)
            #     vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            if "sem_seg" in predictions:
                sem_seg = predictions["sem_seg"].argmax(dim=0).to(self.cpu_device).cpu().numpy()
                # print(sem_seg)
                # print(sem_seg.shape)
                # print(labels)
                # print(areas)
            # 画图：画分割图
                mask_color = colormap[sem_seg].astype(dtype=np.uint8)
                # print(mask_color)
                # cv2.imwrite('output/' + id_frame + "-mask.jpg", mask_color)
                # print(type(mask_color))
                # print(mask_color.shape)
                # print(frame.shape)
                image=np.concatenate((frame, mask_color))
                # cv2.imwrite('output/' + id_frame + "-concat.jpg", image)
                image=cv2.addWeighted(frame,0.3,mask_color,0.7,0)
                # cv2.imwrite('output/' + id_frame + "-add.jpg",image)
                # vis_frame=VisImage(image)
                vis_frame=image
                # vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
                # vis_frame = video_visualizer.draw_sem_seg(
                #     frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                # )
                # print(type(vis_frame))
                # Converts Matplotlib RGB format to OpenCV BGR format

                palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
                # dects = self.tracks[str(cnt+1)]
                # print(id_frame)

            #统计：增加分割结果统计
                labels, areas = np.unique(sem_seg, return_counts=True)
                if int(id_frame) % 300 == 0:
                    for i in range(len(labels)):
                        all_cnt[str(labels[i])] += 1  # 总的计数
                        seg_cnt[str(labels[i])] += 1  # 分割的计数
                        # print(i)
                        # print(i,labels[i],areas[i])
            # 画图：画跟踪框
                cateid=0
                if (self.tracks.get(id_frame)):
                    # id_frame = str(int(id_frame))
                    dects = self.tracks[id_frame]
                    for dect in dects:
                        x1 = int(dect['bbox'][0])
                        x2 = int(dect['bbox'][0]) + int(dect['bbox'][2])
                        y1 = int(dect['bbox'][1])
                        y2 = int(dect['bbox'][1]) + int(dect['bbox'][3])
                        category = categories[dect['category_id']]
                        cateid = dect['cateid']
                        catecnt=all_cnt[dect['category_id']]
                        insid = int(dect['insid'])
                        track_cnt[dect['category_id']] = cateid    # 跟踪的计数
                        # print(track_cnt[dect['category_id']],cateid,insid)
                        color = [int((p * (insid ** 2 - insid + 1)) % 255) for p in palette]
                        color = tuple(color)
                        vis_frame = cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 3)
                        label = '{:s}:{:d}/{:d} '.format(category, cateid, catecnt)
                        # print(label)
                        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                        vis_frame = cv2.rectangle(vis_frame, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color,
                                                  -1)
                        cv2.putText(vis_frame, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2,
                                    [255, 255, 255],
                                    2)
            # 画图：写统计结果
                for i in range(30):
                    # print(categories[str(i)], ins_id_category[str(i)])
                    if(i<19):
                        color=colormap.tolist()[i]
                        color = tuple(color)
                        # color = tuple([int(color[0]*0.7),int(color[1]*0.7),int(color[2]*0.7)])
                    else:
                        color = colormap.tolist()[19]
                        color = tuple(color)
                        # color = tuple([int(color[0]*0.7),int(color[1]*0.7),int(color[2]*0.7)])
                    label = '{:s}:{:d}/{:d} '.format(categories[str(i)],
                                                         seg_cnt[str(i)]+track_cnt[str(i)],
                                                         all_cnt[str(i)])
                    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                    vis_frame = cv2.rectangle(vis_frame, (0, (t_size[1] + 10) * i), (t_size[0] + 1,(t_size[1] + 10)*(i+1)),color, -1) #对角线画矩形框
                    cv2.putText(vis_frame, label, (0, (t_size[1] + 10) * (i+1) - 5), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255],
                                2)

                # vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
                # cv2.imwrite('output/' + id_frame + '-sd.jpg', vis_frame)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    # dects=self.tracks[str(cnt+1-buffer_size)]
                    yield process_predictions(frame, predictions,str(cnt+1-buffer_size))

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                # dects=self.tracks[-len(frame_data):][0]
                yield process_predictions(frame, predictions,str(-len(frame_data)))
        else:
            # for frame in frame_gen:
            predictions=[]
            for cnt, frame in enumerate(frame_gen):
                # dects = self.tracks[str(cnt+1)]
                # print(cnt)
                if cnt%1==0:
                    # print(cnt)
                    predictions=self.predictor(frame)
                yield process_predictions(frame, predictions,str(cnt))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
