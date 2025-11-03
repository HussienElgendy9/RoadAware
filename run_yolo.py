import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as T

from ultralytics import YOLO

# === CONFIG ===
VIDEO_PATH = "Test_Video.mp4"
YOLO_MODEL_PATH = "yolo11n.pt"
ROAD_CLASS_IDX = 0
SEG_INPUT_SIZE = (256, 256)


class DeviceManager:
    @staticmethod
    def get_device():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✅ CUDA Available: {torch.cuda.is_available()}")
        if device == "cuda":
            print(f"🟢 GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("🔴 Running on CPU")
        return device


class ModelLoader:
    def __init__(self, yolo_path, device):
        self.device = device
        self.yolo_model = YOLO(yolo_path)
        self.seg_model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True).eval().to(device)
        if device == "cuda":
            self.seg_model = self.seg_model.half()

    def get_yolo(self):
        return self.yolo_model

    def get_segmentation_model(self):
        return self.seg_model


class RoadSegmenter:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(SEG_INPUT_SIZE),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def segment_road(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
        if self.device == "cuda":
            input_tensor = input_tensor.half()

        with torch.no_grad():
            output = self.model(input_tensor)["out"]

        seg_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
        road_mask = (seg_mask == ROAD_CLASS_IDX).astype(np.uint8) * 255
        road_mask_resized = cv2.resize(road_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        return road_mask_resized


class YOLOTracker:
    def __init__(self, model):
        self.model = model

    def detect(self, frame):
        return self.model.track(frame, persist=True, verbose=False)


class BEVDrawer:
    @staticmethod
    def draw_bev(results, shape):
        bev_map = 255 * np.ones(shape, dtype=np.uint8)
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            center_x = int((xyxy[0] + xyxy[2]) / 2)
            bottom_y = int(xyxy[3])
            cv2.circle(bev_map, (center_x, bottom_y), 4, 0, -1)
        return bev_map


class VideoProcessor:
    def __init__(self, video_path, segmenter, tracker, bev_drawer):
        self.video_path = video_path
        self.segmenter = segmenter
        self.tracker = tracker
        self.bev_drawer = bev_drawer

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            free_space_mask = self.segmenter.segment_road(frame)
            results = self.tracker.detect(frame)
            annotated_frame = results[0].plot()
            bev_map = self.bev_drawer.draw_bev(results, free_space_mask.shape)

            cv2.imshow("YOLOv11n Tracking", annotated_frame)
            cv2.imshow("Free Space Mask", free_space_mask)
            cv2.imshow("BEV Map", bev_map)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()


def main():
    device = DeviceManager.get_device()
    loader = ModelLoader(YOLO_MODEL_PATH, device)

    yolo_model = loader.get_yolo()
    seg_model = loader.get_segmentation_model()

    segmenter = RoadSegmenter(seg_model, device)
    tracker = YOLOTracker(yolo_model)
    bev_drawer = BEVDrawer()

    processor = VideoProcessor(VIDEO_PATH, segmenter, tracker, bev_drawer)
    processor.run()


if __name__ == "__main__":
    main()
