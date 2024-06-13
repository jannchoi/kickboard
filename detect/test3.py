from PIL import Image, ImageDraw
import numpy as np
import os

# 이미지 폴더 경로
image_folder_path = 'v8_5_results/matching/sidewalk/imgs'
label_folder_path = 'v8_5_results/matching/sidewalk/box'
output_folder_path = 'v8_5_results/matching/sidewalk/visualized'  # 결과 이미지를 저장할 폴더

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# 텍스트 파일에서 빨간색 사각형의 좌표를 로드하는 함수
def load_red_coordinates(file_path):
    with open(file_path, 'r') as file:
        coordinates = file.read().strip().split(',')
    # 좌표를 각각의 변수로 분할하여 반환
    x_min, y_min, x_max, y_max = map(int, coordinates)
    return x_min, y_min, x_max, y_max

# 빨간색 사각형을 파란색으로 칠하여 이미지를 저장하는 함수
def draw_blue_rectangle(image, coordinates, output_path):
    draw = ImageDraw.Draw(image)
    x_min, y_min, x_max, y_max = coordinates
    draw.rectangle([x_min, y_min, x_max, y_max], outline="blue", width=3)
    image.save(output_path)

# 빨간색 사각형의 면적을 계산하는 함수
def calculate_red_area(coordinates):
    x_min, y_min, x_max, y_max = coordinates
    width = x_max - x_min
    height = y_max - y_min
    red_area = width * height
    return red_area

# 초록색(보도) 영역의 면적을 계산하는 함수
def calculate_green_area(image, lower_green=(0, 200, 0), upper_green=(100, 255, 100)):
    image_np = np.array(image)
    # 초록색 범위를 조정합니다.
    diff = int(0.15 * 255)
    lower_green = (max(0, lower_green[0] - diff), max(0, lower_green[1] - diff), max(0, lower_green[2] - diff))
    upper_green = (min(255, upper_green[0] + diff), min(255, upper_green[1] + diff), min(255, upper_green[2] + diff))
    
    green_mask = np.all(np.logical_and(image_np >= lower_green, image_np <= upper_green), axis=-1)
    green_area = np.sum(green_mask)
    return green_area

# 초록색(보도) 영역을 시각화하여 이미지를 저장하는 함수
def draw_green_area(image, lower_green=(0, 200, 0), upper_green=(100, 255, 100), output_path=None):
    image_np = np.array(image)
    # 초록색 범위를 조정합니다.
    diff = int(0.15 * 255)
    lower_green = (max(0, lower_green[0] - diff), max(0, lower_green[1] - diff), max(0, lower_green[2] - diff))
    upper_green = (min(255, upper_green[0] + diff), min(255, upper_green[1] + diff), min(255, upper_green[2] + diff))

    green_mask = np.all(np.logical_and(image_np >= lower_green, image_np <= upper_green), axis=-1)
    green_area = np.zeros_like(image_np)
    green_area[green_mask] = [0, 255, 0]  # 초록색으로 설정
    
    output_image = Image.fromarray(green_area)
    
    if output_path:
        output_image.save(output_path)
    
    return output_image

# label_folder_path에 있는 모든 텍스트 파일을 처리
for file_name in os.listdir(label_folder_path):
    if file_name.endswith('.txt'):
        file_path = os.path.join(label_folder_path, file_name)
        red_coordinates = load_red_coordinates(file_path)
        
        # 빨간색 상자의 면적 계산
        red_area = calculate_red_area(red_coordinates)
        
        # 이미지 이름을 가져와서 해당 이미지를 불러옵니다.
        image_name = file_name.replace('_coordinates.txt', '_output.jpg')
        image_path = os.path.join(image_folder_path, image_name)
        image = Image.open(image_path)
        
        # 초록색 면적 계산
        green_area = calculate_green_area(image)
        
        # 파란색 상자를 그린 후 저장
        output_image_path = os.path.join(output_folder_path, image_name.replace('_output.jpg', '_visualized.jpg'))
        draw_blue_rectangle(image, red_coordinates, output_image_path)
        
        # 초록색 면적을 시각화하여 저장
        green_output_image_path = os.path.join(output_folder_path, image_name.replace('_output.jpg', '_green_area.jpg'))
        draw_green_area(image, output_path=green_output_image_path)
        
        # 비율 계산
        ratio = red_area / green_area if green_area != 0 else 0
        
        print(f"파일: {file_name}, sidewalk 면적: {green_area}, 킥보드 면적: {red_area}, 비율: {ratio:.4f}")
