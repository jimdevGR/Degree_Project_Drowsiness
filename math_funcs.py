from math import dist

# define the indices of the eyes and mouth
left_eye_indices = [33, 133, 160, 144, 158, 153]
right_eye_indices = [263, 362, 385, 380, 387, 373]
mouth_indices = [61, 291, 13, 14, 81, 178, 311, 402]

'''
function to calculate the eye-aspect-ratio,
we multiply the coordinates by the width or height of the input image
because they are normalized values by mediapipe between 0-1
'''
def calculate_ear(mediapipe_landmarks, eye_indices, image_width, image_height, side):
    eye_points = {}
    for index in eye_indices:
        eye_x = mediapipe_landmarks[index].x * image_width
        eye_y = mediapipe_landmarks[index].y * image_height
        eye_points[index] = (eye_x, eye_y)
    if side == "Left":
        numenator = dist(eye_points[160], eye_points[144]) + dist(eye_points[158], eye_points[153])
        denominator = 2 * dist(eye_points[33], eye_points[133])
    elif side == "Right":
        numenator = dist(eye_points[385], eye_points[380]) + dist(eye_points[387], eye_points[373])
        denominator = 2 * dist(eye_points[263], eye_points[362])
    ear = numenator / denominator
    return ear

# function to calculate the mouth-aspect-ratio
def calculate_mar(mediapipe_landmarks, mouth_indices, image_width, image_height):
    mouth_points = {}
    for index in mouth_indices:
        mouth_x = mediapipe_landmarks[index].x * image_width
        mouth_y = mediapipe_landmarks[index].y * image_height
        mouth_points[index] = (mouth_x, mouth_y)
    numerator = dist(mouth_points[81], mouth_points[178]) + dist(mouth_points[13], mouth_points[14]) + dist(mouth_points[311], mouth_points[402])
    denominator = 3 * dist(mouth_points[61], mouth_points[291])
    mar = numerator / denominator
    return mar

