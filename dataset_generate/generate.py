
import os
import cv2
import dlib
import numpy as np
from glob import glob

detector = dlib.get_frontal_face_detector()

# modify this variable to where your main project directory is
main_dir = ""

# start_dir points to the dataset of masked individuals
start_dir = main_dir + '/dataset/masked/OneDrive_1_12-27-2020'
pattern   = "*.jpg"

files = []
# no_mask_dir points to the dataset of non-masked individuals
no_mask_dir = main_dir + '/dataset/not_masked/'

# the following variables point to where the images are going to be saved by class
no_shield_dataset_dir = main_dir + '/generated/no_shield/'
shield_dataset_dir = main_dir + '/generated/has_shield/'

p = "shape_predictor_68_face_landmarks.dat"

# Initialize dlib's shape predictor
predictor = dlib.shape_predictor(p)

# shield variable contains the directories of the face shield graphics + the four
# points that will be used for homography.
shields = [
    [
        main_dir + '/Images/face-shield.png',
        np.array([[8, 153], [402, 153], [7, 495], [401, 495]], dtype="float32",)
    ],
    [
        main_dir + '/Images/face-shield2.png',
        np.array([[8, 103], [402, 103], [7, 495], [401, 495]], dtype="float32",)
    ],
    [
        main_dir + '/Images/face-shield3.png',
        np.array([[11, 100], [529, 100], [15, 542], [529, 542]], dtype="float32",)
    ],
    [
        main_dir + '/Images/face-shield4.png',
        np.array([[40, 106], [467, 106], [40, 401], [467, 401]], dtype="float32",)
    ],
]

total_shields = len(shields)

for dir,_,_ in os.walk(start_dir):
    files.extend(glob(os.path.join(dir,pattern)))
    
for i, file in enumerate(files):
    new_f_name = file.split('/OneDrive_1_12-27-2020/')[1].split('_Mask')[0]
    new_f_name = no_mask_dir + '/' + new_f_name + '.png'
    img = cv2.imread(new_f_name)
    masked = cv2.imread(file)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        face = detector(gray)[0]
    except Exception as e:
        continue

    # Get the shape using the predictor
    landmarks=predictor(gray, face)
    
    result = img.copy()
    result = result.astype(np.float32) / 255.0
    resultmasked = masked.copy()
    resultmasked = resultmasked.astype(np.float32) / 255.0
    

    cv2.imwrite(no_shield_dataset_dir + str(i) + '.jpg', result * 255)
    cv2.imwrite(no_shield_dataset_dir + str(i) + '_mask.jpg', resultmasked * 255)

    dst_pts = np.array(
        [
            [landmarks.part(1).x, landmarks.part(19).y],
            [landmarks.part(16).x, landmarks.part(24).y],
            [landmarks.part(1).x, landmarks.part(9).y],
            [landmarks.part(16).x, landmarks.part(9).y],
        ],
        dtype="float32",
    )


    # src_pts = np.array([[8, 153], [402, 153], [7, 495], [401, 495]], dtype="float32",)   # FOR FACESHIELD1 black
    src_pts = shields[i % total_shields][1]   # FOR FACESHIELD2 All Plastic

    dst_pts = np.array(dst_pts, dtype="float32")
    src_pts = np.array(src_pts, dtype="float32")

    # load mask image
    mask_img = cv2.imread(shields[i % total_shields][0], cv2.IMREAD_UNCHANGED)
    mask_img = mask_img.astype(np.float32)
    mask_img = mask_img / 255.0

    # M = cv2.getPerspectiveTransform(src_pts,dst_pts)

    # get the perspective transformation matrix
    M, _ = cv2.findHomography(src_pts, dst_pts)

    # transformed masked image
    transformed_mask = cv2.warpPerspective(
        mask_img,
        M,
        (result.shape[1], result.shape[0]),
        None,
        cv2.INTER_LINEAR,
        cv2.BORDER_CONSTANT,
    )


    # mask overlay
    alpha_mask = transformed_mask[:, :, 3]
    alpha_image = 1.0 - alpha_mask

    for c in range(0, 3):
        result[:, :, c] = (
            alpha_mask * transformed_mask[:, :, c]
            + alpha_image * result[:, :, c]
        )
        resultmasked[:, :, c] = (
            alpha_mask * transformed_mask[:, :, c]
            + alpha_image * resultmasked[:, :, c]
        )

    cv2.imwrite(shield_dataset_dir + str(i) + '.jpg', result * 255)
    cv2.imwrite(shield_dataset_dir + str(i) + '_masked.jpg', resultmasked * 255)
