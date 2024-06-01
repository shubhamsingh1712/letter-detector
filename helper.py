import cv2
import numpy as np
import onnxruntime
class_names =[chr(i) for i in range(65,91)]

rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(26, 3))

def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3):
        mask_img = image.copy()
        det_img = image.copy()

        img_height, img_width = image.shape[:2]
        size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)

        # Draw bounding boxes and labels of detections
        for box, score, class_id in zip(boxes, scores, class_ids):
            #if class_id==1:

            color = colors[class_id]

            x1, y1, x2, y2 = box.astype(int)
            # Draw rectangle
            cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)

            # Draw fill rectangle in mask image
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

            label = class_names[class_id]
            caption = f'{label} {int(score * 100)}%'
            (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=size, thickness=text_thickness)
            th = int(th * 1.2)

            cv2.rectangle(det_img, (x1, y1),
                        (x1 + tw, y1 - th), color, -1)
            cv2.rectangle(mask_img, (x1, y1),
                        (x1 + tw, y1 - th), color, -1)
            cv2.putText(det_img, caption, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

            cv2.putText(mask_img, caption, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

        return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)

class YOLOv8:

    def __init__(self, 
                    path, 
                    conf_thres=0.25, 
                    iou_thres=0.7, 
                    half=True, 
                    padding_percent=None,
                    filter_boxes_by_area=None):

        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.half=half
        self.padding_percent = padding_percent
        self.filter_boxes_by_area = filter_boxes_by_area

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path):
        try:
            self.session = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'])
        except Exception as e:
            print("YOLO model load failed with exception:",e)
            exit()

        # Get model info
        self.get_input_details()
        self.get_output_details()

    def nms(self, boxes, scores, iou_threshold):
        """
        Perform Non-Maximum Suppression (NMS) on the detected bounding boxes.

        NMS is a technique used in object detection tasks to eliminate redundant 
        bounding boxes, keeping only the most likely ones. It works by first sorting 
        the boxes based on their confidence scores and then iteratively selecting the 
        boxes with the highest score while removing overlapping boxes that exceed a 
        certain Intersection over Union (IoU) threshold.

        :param boxes: A numpy array of detected boxes, where each box is represented 
                    by its corner coordinates [x1, y1, x2, y2].
        :param scores: A numpy array of confidence scores corresponding to each box.
        :param iou_threshold: The IoU threshold used to determine when a box should 
                            be suppressed.

        :return: A list of indices of boxes that are kept after applying NMS.
        """
        # Sort the boxes by their confidence scores in descending order.
        sorted_indices = np.argsort(scores)[::-1]

        keep_boxes = []  # Initialize a list to store the indices of boxes to keep.
        while sorted_indices.size > 0:
            # Select the box with the highest score and add its index to the keep list.
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)

            # Calculate the IoU of this box with all other boxes.
            ious = self.compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

            # Find indices of boxes with IoU less than the threshold.
            keep_indices = np.where(ious < iou_threshold)[0]

            # Update the list of indices by removing the suppressed boxes.
            sorted_indices = sorted_indices[keep_indices + 1]

        return keep_boxes


    def compute_iou(self, box, boxes):
        """
        Compute the Intersection over Union (IoU) between a given box and a set of boxes.

        IoU is a metric used to measure the overlap between two bounding boxes. It is 
        calculated as the area of overlap between the boxes divided by the area of union 
        of the boxes. This function computes IoU for a single box against multiple boxes,
        which is commonly used in object detection tasks to measure how much a predicted 
        box overlaps with ground truth boxes.

        :param box: A numpy array representing a single box with coordinates [x1, y1, x2, y2].
        :param boxes: A numpy array of boxes, each represented by coordinates [x1, y1, x2, y2].

        :return: A numpy array containing the IoU values for the given box against each box in `boxes`.
        """
        # Calculate the coordinates of the intersection rectangle.
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])

        # Calculate the area of intersection.
        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

        # Calculate the area of the given box and each box in the array.
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # Calculate the area of union for each comparison.
        union_area = box_area + boxes_area - intersection_area

        # Compute the IoU by dividing the intersection area by the union area.
        iou = intersection_area / union_area

        return iou


    def xywh2xyxy(self, x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y
    
    def detect_objects(self, image):
        """
        Detect objects in the given image, focusing specifically on detecting 'persons'.

        This function processes the input image, performs object detection inference,
        and then filters out all detected objects except those classified as 'person'.

        :param image: An image in which objects (specifically 'persons') are to be detected.

        :return: A tuple of three arrays: boxes (coordinates of detected 'person' objects),
                scores (confidence scores of these detections), and class_ids (class IDs,
                which should all be 0, corresponding to 'person').
        """
        input_tensor = self.prepare_input(image)
        outputs = self.inference(input_tensor)
        self.boxes, self.scores, self.class_ids = self.process_output(outputs)

        return self.boxes, self.scores, self.class_ids

    def add_padding(self, image):
        """
        Add padding to the bounding boxes of an object detection model.

        This method increases the size of each bounding box by a specified percentage
        of its current width and height, enhancing the detection model's robustness 
        by including more context around the detected objects. The padding is added
        equally on all sides of the bounding box.
        padding_percent default is 6%.

        :param image: A numpy array of the image, used to ensure the expanded boxes 
                    don't exceed the image boundaries.
        :return: An updated numpy array of boxes with added padding.
        """

        padding_percent = self.padding_percent/100
        # Calculate padding for width (x-axis) and height (y-axis)
        padding_x = (self.boxes[:, 2] - self.boxes[:, 0]) * padding_percent
        padding_y = (self.boxes[:, 3] - self.boxes[:, 1]) * padding_percent

        # Expand bounding boxes and ensure they stay within image boundaries
        self.boxes[:, 0] = np.maximum(0, self.boxes[:, 0] - padding_x)
        self.boxes[:, 1] = np.maximum(0, self.boxes[:, 1] - padding_y)
        self.boxes[:, 2] = np.minimum(image.shape[1], self.boxes[:, 2] + padding_x)
        self.boxes[:, 3] = np.minimum(image.shape[0], self.boxes[:, 3] + padding_y)

        return self.boxes


    def filter_boxes_by_relative_area(self, image):
        """
        Filter out boxes whose relative area is less than the specified threshold.
        default threshold is 4

        :param boxes: Numpy array of boxes with shape (N, 4), where each box is [x1, y1, x2, y2].
        :param image_shape: A tuple (height, width) representing the dimensions of the image.
        :return: Filtered numpy array of boxes.
        """
        image_height, image_width = image.shape[:2]
        area_image = image_width * image_height
        threshold = self.filter_boxes_by_area

        # Calculate the area of each box and filter
        filtered_boxes, filtered_scores, filtered_classes = [], [], []
        for box, score, class_id in zip(self.boxes, self.scores, self.class_ids):
            x1, y1, x2, y2 = box
            area_box = (x2 - x1) * (y2 - y1)
            relative_area = (area_box / area_image) * 100

            if round(relative_area) >= threshold or (round(relative_area) > threshold//2 and score > 0.4) or (round(relative_area) > 1 and score > 0.6):
                filtered_boxes.append(box)
                filtered_scores.append(score)
                filtered_classes.append(class_id)
            else:
                print(relative_area, score)

        return np.array(filtered_boxes), np.array(filtered_scores), np.array(filtered_classes)
        
    
    def prepare_input(self, image):
        if self.half == True:
            dtype = np.float16
        else:
            dtype = np.float32
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))
        from matplotlib import pyplot as plt
        plt.imshow(input_img)

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(dtype)

        return input_tensor


    def inference(self, input_tensor):
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        return outputs

    def process_output(self, output):
        """
        Process the raw output from the object detection model and extract meaningful information.

        This function filters the predictions based on a confidence threshold, retrieves the
        class IDs of the detected objects with the highest confidence, and extracts their 
        corresponding bounding boxes. It then applies Non-Maximum Suppression (NMS) to remove 
        overlapping bounding boxes.

        :param output: The raw output from the object detection model. This is expected to be 
                    a list where the first element contains the predictions.

        :return: A tuple of three elements: (boxes, scores, class_ids). 
                - boxes: An array of bounding boxes for each detected object.
                - scores: An array of confidence scores for each detection.
                - class_ids: An array of class IDs for each detection.
                If no objects are detected or if the scores are below the confidence threshold,
                empty lists are returned for each.
        """
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below the specified confidence threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        # If no scores are above the threshold, return empty lists
        if len(scores) == 0:
            return np.array([]), np.array([]), np.array([])

        # Identify the class of each object with the highest confidence score
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Extract bounding boxes for each object detection
        boxes = self.extract_boxes(predictions)

        # Apply Non-Maximum Suppression to refine the list of bounding boxes
        indices = self.nms(boxes, scores, self.iou_threshold)

        # Return the filtered and refined bounding boxes, scores, and class IDs
        return boxes[indices], scores[indices], class_ids[indices]


    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = self.xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):
        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes


    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):

        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)
