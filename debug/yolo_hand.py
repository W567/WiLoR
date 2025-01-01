import pyrealsense2 as rs
from dataclasses import dataclass
import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import Union
import os
import supervision as sv
BOX_ANNOTATOR = sv.BoxAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()


@dataclass
class DetectionResult:
    label: str
    rect: np.ndarray
    score: float
    state: str = "N"
    object_rect: Union[np.ndarray, None] = None


class DetectionModelBase(ABC):
    @abstractmethod
    def predict(self, detection_results, im, vis_im):
        pass


class YoloHandModel(DetectionModelBase):
    def __init__(
        self,
        threshold: float = 0.5,
        margin: int = 10,
        device: str = "cuda:0",
    ):
        self.threshold = threshold
        self.margin = margin
        self.device = device

        # init model
        from ultralytics import YOLO

        # get the path to this file
        path = os.path.dirname(os.path.abspath(__file__))
        self.detector = YOLO("../pretrained_models/detector.pt")
        self.detector = self.detector.to(device)

    def predict(self, im):
        # im : BGR image
        results = self.detector(im, conf=0.5, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        xyxys = [xyxy for xyxy in detections.xyxy]
        scores = detections.confidence.tolist()
        hand_detections = []
        for i, detection in enumerate(detections):
            label = "left_hand" if detection[5]['class_name'] == 'left' else "right_hand"
            hand_detection = DetectionResult(
                label=label,
                rect=np.array([xyxys[i][0], xyxys[i][1], xyxys[i][2], xyxys[i][3]]),
                score=scores[i],
            )
            hand_detections.append(hand_detection)

        # visualize
        labels = [self.detector.names[i] for i in detections.class_id]
        labels_with_scores = [f"{label} {score:.2f}" for label, score in zip(labels, scores)]
        visualization = im.copy()
        visualization = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
        visualization = BOX_ANNOTATOR.annotate(scene=visualization, detections=detections)
        visualization = LABEL_ANNOTATOR.annotate(
            scene=visualization, detections=detections, labels=labels_with_scores
        )
        return hand_detections, visualization




def main():
    # Initialize the YOLO hand model
    model = YoloHandModel()

    # Configure the RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Adjust resolution and framerate as needed

    # Start streaming
    pipeline.start(config)

    try:
        while True:
            # Wait for a frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert RealSense frame to numpy array
            color_image = np.asanyarray(color_frame.get_data())
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # Run the YOLO hand detection model
            body_detections, visualization = model.predict(color_image)

            # Display the detection results
            cv2.imshow("YOLO Hand Detection", visualization)

            # Exit the loop if the user presses 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop streaming and release resources
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

