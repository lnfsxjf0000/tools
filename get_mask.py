import cv2
import numpy as np
import json
import os.path as path
import os
import random
# from tqdm import tqdm

def cv_show(img, window_name="img", win_x=300, win_y=600, f_x=1, f_y=1, wait_flag=0,BGR_flag=1):
    if(BGR_flag==0):
        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    de_img = cv2.resize(img,(0,0), fx=f_x, fy=f_y, interpolation=cv2.INTER_CUBIC)  # 指定参数  xx倍缩放
    cv2.imshow(window_name, de_img)
    cv2.moveWindow(window_name, win_x, win_y)
    if wait_flag != 0:
        cv2.waitKey(0)


#在json数据根目录执行本脚本


separator=os.sep

data_dir='label'#from where to replace and mkdir mask

label_map={"impurity|notSure":244,
           "graphite-edgeCrinkle":180}





root=os.getcwd()
def json2mask(file_name:str,
              output_path:str,
              label_map
              ):

    json_file = json.load(open(file_name))
    img_h, img_w, img_name = json_file["imageHeight"], json_file["imageWidth"], json_file["imagePath"]
    json_shapes = json_file["shapes"]
    shapes_num = len(json_shapes)
    img_mask = np.zeros((img_h, img_w, 3), np.uint8)

    for i in range(shapes_num):
        label = json_shapes[i]['label']
        points_num = len(json_shapes[i]['points'])
        points = []
        for j in range(points_num):
            points.append(json_shapes[i]['points'][j])
        points = np.array(points, dtype=np.int32)

        mask_color = []
        for n in range(3):
            # mask_color.append(random.randint(0, 255))
            mask_color.append(label_map[label])
        cv2.fillConvexPoly(img_mask, points, mask_color)


    mask_file = file_name.split(separator)[-1].split('.')[0] + '_mask.png'
    output_mask = path.join(output_path, mask_file)
    print("input: ", file_name)
    print("output: ",output_mask)

    # 是否输出单通道图像
    cv2.imwrite(output_mask, img_mask[:, :, 0])  # 第三个参数压缩 注意
    # cv_show(img_mask,"mask",win_x=100,win_y=100,f_x=0.2,f_y=0.2,wait_flag=1)




if __name__ == "__main__":
    count=0
    for parent, dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.split('.')[-1] != 'json':
                continue
            file_name = path.join(parent, filename)
            outdir = parent.replace(data_dir, 'mask')
            os.makedirs(outdir, exist_ok=True)
            count += 1
            json2mask(file_name, outdir, label_map)
            print("count: ",count)
            print(" ")

    print(" json2mask  finished!")