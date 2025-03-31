import os
import cv2
import json
import numpy as np
from datetime import timedelta
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Initialize once (outside the loop) - load it globally for efficiency
ppocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

# ===== 2. IMPROVED TEXT EXTRACTION =====
def extract_text_ppocr(image, bbox):
    """Extract text using PaddleOCR from a bounding box"""
    try:
        x1, y1, x2, y2 = bbox
        # Ensure valid coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        
        # Run PaddleOCR
        result = ppocr.ocr(roi, cls=True)
        
        # Extract and combine text
        if result and result[0]:
            texts = []
            for line in result[0]:
                if len(line) >= 2:  # Should have bbox and text with confidence
                    text_info = line[1]  # This contains (text, confidence)
                    if isinstance(text_info, tuple) and len(text_info) >= 1:
                        texts.append(text_info[0])
                    elif isinstance(text_info, str):
                        texts.append(text_info)
            
            combined_text = ' '.join(texts)
            return combined_text.strip() if combined_text else None
        return None
    except Exception as e:
        print(f"PPOCR failed: {str(e)}")
        return None


# ===== 3. NON-MAXIMUM SUPPRESSION FOR DUPLICATE DETECTION =====
def apply_nms(elements, iou_threshold=0.5):
    """Apply non-maximum suppression to remove overlapping detections"""
    if not elements:
        return []
    
    # Sort by confidence
    elements = sorted(elements, key=lambda x: x['confidence'], reverse=True)
    final_elements = []
    
    for element in elements:
        # Skip if already merged/suppressed
        if element.get('suppressed', False):
            continue
            
        final_elements.append(element)
        
        # Mark overlapping boxes as suppressed
        for other in elements:
            if other.get('suppressed', False) or other is element:
                continue
                
            # Calculate IoU
            box1 = np.array(element['position'])
            box2 = np.array(other['position'])
            
            # Area of intersection
            x_left = max(box1[0], box2[0])
            y_top = max(box1[1], box2[1])
            x_right = min(box1[2], box2[2])
            y_bottom = min(box1[3], box2[3])
            
            if x_right < x_left or y_bottom < y_top:
                continue
                
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            
            # Areas of each box
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            
            # IoU calculation
            iou = intersection_area / float(box1_area + box2_area - intersection_area)
            
            # If boxes overlap significantly and are the same class
            if iou > iou_threshold and element['name'] == other['name']:
                other['suppressed'] = True
    
    return [e for e in final_elements if not e.get('suppressed', False)]

# ===== 4. FOCUS STATE RESOLVER =====
def resolve_focus_conflicts(elements):
    """Ensure only one element can be focused at a time"""
    if not elements:
        return elements
        
    # Get all focused elements
    focused_elements = [e for e in elements if e['state'] == 'focused']
    
    # If more than one element is focused, keep only the highest confidence one
    if len(focused_elements) > 1:
        # Sort by confidence
        focused_elements.sort(key=lambda x: x['confidence'], reverse=True)
        # Keep the highest confidence
        highest_conf = focused_elements[0]
        
        # Set others to unfocused
        for element in elements:
            if element['state'] == 'focused' and element is not highest_conf:
                element['state'] = 'unfocused'
                # Update name by removing "_focused" suffix if present
                if "_focused" in element['name']:
                    element['name'] = element['name'].replace("_focused", "")
    
    return elements

# Calculate IoU between two boxes
def calculate_iou(box1, box2):
    """Calculate Intersection over Union of two boxes"""
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate area of each box
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate IoU
    iou = intersection_area / float(area1 + area2 - intersection_area)
    return iou

# NEW FUNCTION: Get incremental filename to avoid overwriting
def get_incremental_filename(base_path):
    """Create an incremental filename if file already exists"""
    if not os.path.exists(base_path):
        return base_path
        
    # Split into name and extension
    base_dir = os.path.dirname(base_path)
    filename = os.path.basename(base_path)
    name_parts = os.path.splitext(filename)
    base_name = name_parts[0]
    extension = name_parts[1]
    
    counter = 1
    while True:
        new_filename = f"{base_name}_{counter}{extension}"
        new_path = os.path.join(base_dir, new_filename)
        if not os.path.exists(new_path):
            return new_path
        counter += 1

# ===== 5. COMBINED PROCESSING FUNCTION =====
def process_video_with_output(video_path, model_path, output_json_path, output_video_path=None, confidence_threshold=0.9):
    """Process video with model, generate JSON log and annotated video output"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Ensure we don't overwrite existing files
    output_json_path = get_incremental_filename(output_json_path)
    if output_video_path:
        output_video_path = get_incremental_filename(output_video_path)
    
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise IOError("Could not open video file")
    
    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer if output video is requested
    video_writer = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # State tracking
    state_log = []
    previous_elements = []
    frame_count = 0
    
    print(f"Processing video with {total_frames} frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Processing frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
        
        # Get accurate timestamp
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp = str(timedelta(milliseconds=timestamp_ms)).split(".")[0]
        
        # Run detection
        results = model(frame)[0]
        current_elements = []
        
        for box in results.boxes:
            if box.conf > confidence_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls)
                class_name = results.names[cls_id]
                
                element = {
                    "type": class_name.split('_')[0],
                    "name": class_name,
                    "position": [x1, y1, x2, y2],
                    "state": "focused" if "_focused" in class_name else "unfocused",
                    "confidence": float(box.conf),
                    "text": None  # Will be filled later
                }
                
                current_elements.append(element)
        
        # Apply NMS to filter duplicates
        current_elements = apply_nms(current_elements, iou_threshold=0.5)
        
        # Resolve focus conflicts
        current_elements = resolve_focus_conflicts(current_elements)
        
        # Extract text for each element
        for element in current_elements:
            if element['type'] in ('button', 'input'):
                element['text'] = extract_text_ppocr(frame, element['position'])
        
        # ===== IMPROVED CHANGE DETECTION WITH RULES =====
        changed = False
        
        # First frame always gets recorded
        if not previous_elements:
            changed = True
        # Different number of elements is definitely a change
        elif len(current_elements) != len(previous_elements):
            changed = True
        else:
            # Compare elements when counts match
            # Sort by position for stable comparison
            curr_sorted = sorted(current_elements, key=lambda x: (x['position'][0], x['position'][1]))
            prev_sorted = sorted(previous_elements, key=lambda x: (x['position'][0], x['position'][1]))
            
            for curr, prev in zip(curr_sorted, prev_sorted):
                # Check if element type/name changed
                if curr['name'] != prev['name']:
                    changed = True
                    break
                    
                # Check if focused state changed (important)
                if curr['state'] != prev['state']:
                    changed = True
                    break
                
                # For position, we check if there's NO overlap (completely different position)
                iou = calculate_iou(curr['position'], prev['position'])
                if iou < 0.1:  # Less than 10% overlap means significantly different position
                    changed = True
                    break
        
        # Add to log if changed or first frame
        if changed or not state_log:
            state_log.append({
                "timestamp": timestamp,
                "ui_elements": current_elements
            })
            # Create a deep copy to avoid reference issues
            previous_elements = []
            for elem in current_elements:
                previous_elements.append(elem.copy())
        
        # Draw annotations on frame if video output is requested
        if video_writer:
            # Draw current frame number and timestamp
            cv2.putText(
                frame,
                f"Frame: {frame_count}, Time: {timestamp}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            
            # Draw detection boxes
            for element in current_elements:
                x1, y1, x2, y2 = element['position']
                
                # Use different colors for different states
                color = (0, 255, 0) if element['state'] == 'unfocused' else (0, 0, 255)
                
                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Prepare label with class, state and confidence
                label = f"{element['name']} {element['confidence']:.2f}"
                
                # Add text if available
                if element['text']:
                    label += f" | '{element['text']}'"
                
                # Draw label background
                cv2.rectangle(
                    frame,
                    (x1, y1 - 20),
                    (x1 + len(label) * 8, y1),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
            
            video_writer.write(frame)
    
    # Release resources
    cap.release()
    if video_writer:
        video_writer.release()
    
    # Write JSON file
    with open(output_json_path, 'w') as f:
        json.dump(state_log, f, indent=2, default=str)
    
    print(f"Processing complete. Generated {len(state_log)} state entries.")
    print(f"JSON saved to: {output_json_path}")
    if output_video_path:
        print(f"Annotated video saved to: {output_video_path}")

# ===== 6. MAIN EXECUTION =====
if __name__ == "__main__":
    try:
        VIDEOS_DIR = os.path.join('.', 'videos')
        JSON_DIR = os.path.join('.', 'json')
        video_path = os.path.join(VIDEOS_DIR, 'stream.mp4')
        json_path = os.path.join(JSON_DIR, '')
        model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')
        
        # Generate output paths
        vid_base_name = os.path.splitext(video_path)[0]
        json_base_name = os.path.splitext(json_path)[0]
        output_json_path = f"{json_base_name}_ui_log_ppocr.json"
        output_video_path = f"{vid_base_name}_annotated_ppocr.mp4"
        
        process_video_with_output(
            video_path=video_path,
            model_path=model_path,
            output_json_path=output_json_path,
            output_video_path=output_video_path,
            confidence_threshold=0.9  # Set to 0.9 as required
        )
    except Exception as e:
        print(f"Error: {str(e)}")