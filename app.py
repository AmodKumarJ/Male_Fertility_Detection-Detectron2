import streamlit as st
import cv2
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import os
import csv

setup_logger()

import cv2
import pandas as pd



def load_annotation_file(file_path):
    annotations = pd.read_csv(file_path)
    return annotations
# Calculate Euclidean distance between two points
def calculate_distance(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
def load_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = "model_final (5).pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    class_names = ["impurity1", "sperm", "impurity2"]  # Replace with your actual class names
    MetadataCatalog.get("my_dataset_val").set(thing_classes=class_names)
    predictor = DefaultPredictor(cfg)
    return predictor, cfg
def visualize_annotations(video_file, annotations,sperm_count, movement_threshold=5):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        st.write("Error: Unable to open video file")
        return

    frame_count = 0
    moving_objects_count = 0  # Initialize count of moving objects (sperm)

    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        st.write("Error: Unable to read the first frame")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate absolute difference between the current frame and the previous frame
        frame_diff = cv2.absdiff(prev_frame, frame)
        # Convert the difference to grayscale
        frame_diff_gray = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        # Apply thresholding to obtain binary image
        _, thresh = cv2.threshold(frame_diff_gray, 30, 255, cv2.THRESH_BINARY)
        # Find contours in the binary image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize a flag to track if any moving object is detected in this frame
        moving_object_detected = False

        # Filter annotations for the class "sperm"
        frame_annotations = annotations[annotations['Frame'] == frame_count]
        sperm_annotations = frame_annotations[frame_annotations['Class'] == 'sperm']

        # Draw bounding boxes for each "sperm" annotation
        for _, annotation in sperm_annotations.iterrows():
            x1, y1, x2, y2 = int(annotation['X0']), int(annotation['Y0']), int(annotation['X1']), int(annotation['Y1'])

            # Check if the annotation represents a moving object
            if any(cv2.pointPolygonTest(contour, (int((x1 + x2) / 2), int((y1 + y2) / 2)), False) >= 0 for contour in contours):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                moving_objects_count += 1  # Increment count for each detected sperm
                moving_object_detected = True

        # Update the previous frame
        prev_frame = frame.copy()

        # Display the frame with annotations

        frame_count += 1
        # print("frame no:",frame_count)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Print the count of moving objects detected
    # st.write(f"Total number of moving objects (sperm) detected: {moving_objects_count}")
    st.write(f"<p style='font-size: 24px; color: #ffffff;'>{(moving_objects_count/sperm_count)*100:.2f}% of sperms are active and showing good movement</p>", unsafe_allow_html=True)

predictor, cfg = load_model()

st.title(': THE MALE FERTILITY DETECTION :')

uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi"])

if uploaded_file is not None:
    # Add fields for sample taken and diluent added
    sample_taken_ml = st.number_input("Sample Taken (ml)", min_value=0, step=1)
    diluent_added_ml = st.number_input("Diluent Added (ml)", min_value=0, step=1)

    if uploaded_file.type not in ["video/mp4", "video/avi"]:
        st.error("Please upload an MP4 or AVI video file.")
    elif sample_taken_ml == 0:
        st.error("Please enter the amount of sample taken.")
    elif diluent_added_ml == 0:
        st.error("Please enter the amount of diluent added.")
    else:
        if uploaded_file is not None:
            # Temporary directory to save processed frames and video
            temp_dir = "framesAndvideo"
            output_frames_dir = os.path.join(temp_dir, "output_frames")
            os.makedirs(output_frames_dir, exist_ok=True)
            output_video_dir = os.path.join(temp_dir, "output_video")
            os.makedirs(output_video_dir, exist_ok=True)

            video_path = os.path.join(temp_dir, "uploaded_video.mp4")

            # Save uploaded video to temporary directory
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())

            # Open the video file
            video_capture = cv2.VideoCapture(video_path)
            video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(video_capture.get(cv2.CAP_PROP_FPS))

            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define video codec (XVID is commonly supported)
            output_video_path = os.path.join(output_video_dir, "output_video.avi")
            output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (video_width, video_height))

            st.write("Processing video...")
            progress_bar = st.progress(0)

            category_counts = {}

            # Open CSV file to save annotations
            annotation_file_path = os.path.join(temp_dir, "annotations.csv")
            with open(annotation_file_path, mode='w', newline='') as annotation_file:
                csv_writer = csv.writer(annotation_file)
                csv_writer.writerow(["Frame", "Class", "Confidence", "X0", "Y0", "X1", "Y1"])

                for frame_number in range(frame_count):
                    progress = (frame_number + 1) / frame_count
                    progress_bar.progress(progress)

                    # Read frame from the video
                    ret, frame = video_capture.read()
                    if not ret:
                        break

                    # Perform object detection on the frame
                    outputs = predictor(frame[:, :, ::-1])

                    # Counting the number of detected objects by category
                    pred_classes = outputs["instances"].pred_classes.to("cpu").numpy()

                    # Update category counts
                    for category_id in pred_classes:
                        category_name = MetadataCatalog.get("my_dataset_val").thing_classes[category_id]
                        if category_name not in category_counts:
                            category_counts[category_name] = 1
                        else:
                            category_counts[category_name] += 1

                    # Draw bounding boxes and labels on the frame
                    v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get("my_dataset_val"), scale=1.0)
                    instances = outputs["instances"].to("cpu")
                    v = v.draw_instance_predictions(instances)

                    # Save the frame with annotated bounding boxes
                    output_frame_path = os.path.join(output_frames_dir, f"frame_{frame_number:05d}.jpg")
                    cv2.imwrite(output_frame_path, v.get_image()[:, :, ::-1])

                    # Write annotations to the CSV file
                    for i, box in enumerate(instances.pred_boxes.tensor.numpy()):
                        x0, y0, x1, y1 = box
                        confidence = instances.scores[i].item()
                        class_id = instances.pred_classes[i].item()
                        class_name = MetadataCatalog.get("my_dataset_val").thing_classes[class_id]
                        csv_writer.writerow([frame_number, class_name, confidence, x0, y0, x1, y1])

                    # Write the processed frame to the output video
                    output_video.write(v.get_image()[:, :, ::-1])

            # Output category counts
            st.write("<h3 style='color: #3366ff;'>Category Counts:</h3>", unsafe_allow_html=True)
            sperm_count=0
            total_objects = sum(category_counts.values())
            for category_name, count in category_counts.items():
                st.write(f"<p>Category '<span style='font-weight: bold;'>{category_name}</span>': {count / total_objects:.2%} of total</p>", unsafe_allow_html=True)
                if category_name=="sperm":
                    sperm_count = count
                    # st.write(f"<p style='font-weight: bold;'>Sperm Count: {sperm_count}</p>", unsafe_allow_html=True)
                # st.write(f"Category '{category_name}'  Count: {count / frame_count:.2f}")


            # calculate the sperms from counts and diluents
            if sample_taken_ml > 0:
                total_count = (sperm_count/frame_count) * ((sample_taken_ml + diluent_added_ml) / (sample_taken_ml / (sample_taken_ml + diluent_added_ml))) * 1000
                st.write(f"{total_count // sample_taken_ml}/ml are detected ")
            else:
                st.error("Sample taken (ml) should be greater than 0.")

            # Release resources
            video_capture.release()
            output_video.release()
            cv2.destroyAllWindows()

            # Display download  link for the annotation file

            st.write("<h3 style='color: #3366ff;'>Video processing complete!</h3>", unsafe_allow_html=True)
            st.write("<h3>Applying CSR-DCF algorithm</h3>", unsafe_allow_html=True)
            annotations = load_annotation_file(annotation_file_path)

            # Visualize annotations on frames, counting only annotations with class "sperm"
            visualize_annotations(video_path, annotations,sperm_count)