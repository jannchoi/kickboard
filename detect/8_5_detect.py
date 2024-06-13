import os
import cv2
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

def parse_txt_file(txt_file):
    class_coords = defaultdict(list)
    with open(txt_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            coordinates = [float(x) for x in parts[1:]]
            class_coords[class_id].append(coordinates)
    return class_coords

def extract_class_colors(image, coordinates):
    colors = []
    for coord_set in coordinates:
        xs = coord_set[::2]
        ys = coord_set[1::2]
        for x, y in zip(xs, ys):
            pixel_color = image[int(y * image.shape[0]), int(x * image.shape[1])]
            colors.append(pixel_color)
    return colors

def find_representative_color(colors):
    colors = np.array(colors)
    representative_color = np.median(colors, axis=0)
    return tuple(representative_color.astype(int))

def match_classes_with_colors(txt_folder, image_folder):
    # 숫자와 문자열 매핑 정의
    class_mapping = {0: 'bic_road', 1: 'curb', 2: 'kickboard', 3: 'lane', 4: 'road', 5: 'sidewalk', 6: 'verge'}

    for txt_file in os.listdir(txt_folder):
        if txt_file.endswith('.txt'):
            base_name = os.path.splitext(txt_file)[0]
            image_file = os.path.join(image_folder, base_name + '.jpg')

            if not os.path.exists(image_file):
                print(f"Image file for {txt_file} not found, skipping.")
                continue

            class_coords = parse_txt_file(os.path.join(txt_folder, txt_file))
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_height, image_width, _ = image.shape

            class_representative_colors = {}
            for class_id, coords in class_coords.items():
                colors = extract_class_colors(image, coords)
                representative_color = find_representative_color(colors)
                class_representative_colors[class_id] = representative_color

            # 이미지 색칠 및 클래스명 쓰기
            colorized_image = colorize_image(image, class_representative_colors)
            image_with_text, class_2_coords = add_class_names(colorized_image, class_coords)

            # 그리드 라인 추가 및 사각형 그리기
            image_with_grid = add_grid_lines_and_bbox(image_with_text, class_2_coords)
            
            # 파란색 박스 안에서 클래스 '2'를 제외한 가장 많은 숫자 출력
            most_common_number = find_most_common_number(class_coords, class_2_coords, image_width, image_height)
            if most_common_number is not None:
                most_common_object = class_mapping.get(most_common_number, "Unknown")      
            else:
                most_common_object = "None"
            print(f"The kickboard for {base_name}: is in {most_common_object}")

            # 결과 이미지 출력
            plt.imshow(image_with_grid)
            plt.axis('off')
            plt.title(base_name)
            plt.show()

def colorize_image(image, class_colors):
    for class_id, color in class_colors.items():
        mask = np.all(image == [0, 0, 0], axis=-1)
        image[mask] = color

    return image

def add_class_names(image, class_coords):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    font_thickness = 1
    text_color = (255, 255, 255)
    special_text_color = (255, 0, 0)  # Blue color for class '2'
    class_2_coords = []

    for class_id, coords in class_coords.items():
        for coord_set in coords:
            xs = coord_set[::2]
            ys = coord_set[1::2]
            for x, y in zip(xs, ys):
                x = int(x * image.shape[1])
                y = int(y * image.shape[0])
                text_x = x
                text_y = y
                if class_id == 2:
                    cv2.putText(image, str(class_id), (text_x, text_y), font, font_scale, special_text_color, font_thickness)
                    class_2_coords.append((text_x, text_y))
                else:
                    cv2.putText(image, str(class_id), (text_x, text_y), font, font_scale, text_color, font_thickness)

    return image, class_2_coords

def add_grid_lines_and_bbox(image, class_2_coords, grid_size=32):
    image_with_grid = image.copy()
    h, w, _ = image_with_grid.shape
    # Draw horizontal lines
    for y in range(0, h, grid_size):
        cv2.line(image_with_grid, (0, y), (w, y), (255, 255, 255), 1)
    # Draw vertical lines
    for x in range(0, w, grid_size):
        cv2.line(image_with_grid, (x, 0), (x, h), (255, 255, 255), 1)
    
    # Draw bounding box around class 2 coordinates
    if class_2_coords:
        x_coords, y_coords = zip(*class_2_coords)
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        # Set bounding box height to 30% of the original height and position it at the bottom
        new_height = int((y_max - y_min) * 0.3)
        y_max_new = y_max
        y_min_new = y_max_new - new_height
        cv2.rectangle(image_with_grid, (x_min, y_min_new), (x_max, y_max_new), (255, 0, 0), 2)

    return image_with_grid

def find_most_common_number(class_coords, class_2_coords, image_width, image_height):
    if not class_2_coords:
        return None

    x_coords, y_coords = zip(*class_2_coords)
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    counts = Counter()
    for class_id, coords_list in class_coords.items():
        if class_id == 2:
            continue
        for coords in coords_list:
            xs = coords[::2]
            ys = coords[1::2]
            for x, y in zip(xs, ys):
                x_pixel = int(x * image_width)
                y_pixel = int(y * image_height)
                if x_min <= x_pixel <= x_max and y_min <= y_pixel <= y_max:
                    counts[class_id] += 1

    if counts:
        most_common = counts.most_common(1)[0][0]
    else:
        most_common = None

    return most_common

# 폴더 경로
txt_folder = 'v8_5_results/labels'
image_folder = 'v8_5_results/images'

# 클래스와 대표 색상 매칭 및 이미지 색칠 및 출력
match_classes_with_colors(txt_folder, image_folder)
