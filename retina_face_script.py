import os
import cv2
from retinaface import RetinaFace

# Path to your dataset directory
dataset_path = '/home/msbalquin/AgeDB/'
# images = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Iterate through each image in the dataset
# for img_path in images:
    # Load the image
img_path = '/home/msbalquin/AgeDB/128_Welles_4799_A_22.jpg'
img = cv2.imread(img_path)
# Detect faces
faces = RetinaFace.detect_faces(img_path, align = True)

landmarks = result["face_1"]["landmarks"]
left_eye = landmarks["left_eye"]
right_eye = landmarks["right_eye"]
nose = landmarks["nose"]

img_aligned = postprocess.alignment_procedure(img, right_eye, left_eye, nose)
img_aligned = img_aligned[:,:,::-1]

faces = RetinaFace.detect_faces(img_path, align = True)

##sudo code
bounding box -> square bounding box -> crop image to bounding box -> resize image to 112x112 -> write image

# print(type(faces))
# Save or process the faces as needed
for i, face in enumerate(faces):
    # Example: Save each extracted face as an image
    face_path = os.path.join(dataset_path, f'z{i}.jpg')
    # face_path = os.path.join(dataset_path, f'extracted_face.jpg')
    cv2.imwrite(str(face_path), face)
