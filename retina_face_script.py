import os
import cv2
from retinaface import RetinaFace
from retinaface.commons import postprocess
import math
import argparse

def is_image(filename):
    return os.path.splitext(filename)[1].lower() in image_extensions

def align_and_resize(img_path, out_path):
    result = RetinaFace.detect_faces(img_path)
    if len(result) == 0:
        return 0
    
    landmarks = result["face_1"]["landmarks"]
    left_eye = landmarks["left_eye"]
    right_eye = landmarks["right_eye"]
    nose = landmarks["nose"]

    img = cv2.imread(img_path)
    img_aligned = postprocess.alignment_procedure(img, right_eye, left_eye, nose)
    img_aligned = img_aligned[0]
    im_width = img_aligned.shape[1]
    im_height = img_aligned.shape[0]

    results2 = RetinaFace.detect_faces(img_path)
    if len(results2) == 0:
        return 0

    coords = results2['face_1']["facial_area"]
    print(coords)

    width = coords[2]-coords[0]
    height= coords[3]-coords[1]

    w_diff = width-height
    h_diff = height-width

    if width < height:
        x1_new = math.floor(coords[0] - h_diff/2)
        x2_new = math.floor(coords[2] + h_diff/2)

        y1_new = coords[1]
        y2_new = coords[3]
    else:
        x1_new = coords[0]
        x2_new = coords[2] 

        y1_new = math.floor(coords[1] + w_diff/2)
        y2_new = math.floor(coords[3] + w_diff/2)

    # print(f'BEFORE: (x1,x2) (y1,y2) = ({x1_new},{x2_new}) ({y1_new},{y2_new})')
    if x1_new < 0:
        x2_new += abs(x1_new)
        x1_new = 0
    elif x2_new >= im_width:
        diff = x2_new - im_width
        x1_new -= diff
        x2_new = im_width-1
    elif y1_new < 0:
        y2_new += abs(y1_new)
        y1_new = 0
    elif y2_new >= im_height:
        diff = y2_new - im_height
        y1_new -= diff
        y2_new = im_height-1

    # print(f'AFTER: (x1,x2) (y1,y2) = ({x1_new},{x2_new}) ({y1_new},{y2_new})')
    new_image = img_aligned[y1_new:y2_new, x1_new:x2_new, :]

    new_image = cv2.resize(new_image, (112,112))

    cv2.imwrite(str(out_path), new_image)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Align faces in images")
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        required=True,
        help="Directory containing the images to be aligned",
        dest="input_dir",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        required=True,
        help="Directory where the aligned images will be saved",
        dest="output_dir",
    )
    # Path to your dataset directory
    # images = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    args = parser.parse_args()
    root_dir = args.input_dir
    out_dir = args.output_dir
    total = 0
    image_extensions = {'.jpg', '.jpeg', '.png'} # could add more for diff image types
    print("=== Beginning to parse ===")
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if is_image(file):
                # Construct full file path
                # print(f'subdir {os.path.relpath(file_path, start=subdir)}')
                file_path = os.path.join(subdir, file)
                relative_path = os.path.relpath(subdir, start=root_dir)
                out_folder = os.path.join(out_dir,relative_path)
                # print(f'out_path {out_folder}')
                if not os.path.exists(out_folder):
                    try: 
                        os.mkdir(out_folder)
                    except FileExistsError as e:
                        print(f'Recieved FileExistsError{e}')
                        
                out_path = os.path.join(out_folder,file)
                if os.path.exists(out_path):
                    print(f'Already parsed {out_path}')
                    continue
                # print(f'file_path: {file_path}, out_path: {out_path}')
                # Load and process the image
                align_and_resize(file_path,out_path)
                total +=1
                if total % 200:
                    print(f'total parsed {total}!')
            else:
                print(f"Failed to read image: {file_path}")

# for each image in directory
    
# Detect faces