import numpy as np
import cv2
import os,time
from tqdm import tqdm
from keras.models import load_model
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import torch
from PIL import Image
from torchvision import transforms
map_dict={0:'black',1:'no_black'}
device = torch.device("cuda: 0")


"""
加载torch模型并预测
"""


def load_torch_model(model_path):
    model = torch.load(model_path)
    model.eval()
    model.to(device)
    with torch.no_grad():
        return model

trans=transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def process_img(frame):
    img = cv2.resize(frame,(224,224))
    img = np.asarray(img[:, :, (2, 1, 0)], dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def detect_video_path(video_path,model_path):
    for video_name in tqdm(os.listdir(video_path)):
        if video_name.endswith('mp4'):
            video_full_path = os.path.join(video_path,video_name)
            print(video_full_path)
            detect_video(video_full_path,model_path)


def detect_video(video_path,model_path):
    videoCapture = cv2.VideoCapture(video_path)
    fps = int(videoCapture.get(cv2.CAP_PROP_FPS))
    if model_path.endswith('pth'):  # 加载pth模型与h5模型
        model = load_torch_model(model_path)
    else:
        model = load_model(model_path)
    print('loading keras_tf...')
    w = videoCapture.get(3)
    h = videoCapture.get(4)
    print('w,h',w,h)
    print('fps:',fps)
    fourcc_1 = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # opencv3.0
    size = (int(w),int(h))
    # 指定写视频的格式, I420-avi, MJPG-mp4
    dirs,name = os.path.split(video_path)
    write_video_path = os.path.join(dirs, name.split('.')[0]+'_out_pt.avi')
    print('写入视频路劲：',write_video_path)
    videoWriter = cv2.VideoWriter(write_video_path, fourcc_1, int(fps), size)
    frame_count=0
    while True:
        start_time = time.time()
        # print('当前帧:', fream_count)
        rval, frame = videoCapture.read()
        if rval is False:
            print('video is over')
            break
        if frame_count%(int(fps)*7)==0:
            im = Image.fromarray(np.uint8(frame[:,:,[2,1,0]]))
            image = trans(im)
            image = image.unsqueeze(0)
            # img = process_img(frame)
            # pre = keras_tf.predict(img)
            # image = np.resize(frame, (224, 224, 3))
            # image = ((np.asarray(image) / 255.0) - np.array([0.485, 0.456, 0.406])) #/ # np.array([0.229, 0.224, 0.225])
            # image = np.expand_dims(np.transpose(image, axes=[2, 0, 1]), 0)
            # pre = keras_tf(torch.from_numpy(image).float().to(device))
            pre = model(image.float().to(device))
            pre_index = np.argmax(pre.cpu().detach().numpy())
            #         print(cate,cate_map[np.argmax(pre[0])])
            label = map_dict[pre_index]
            cv2.putText(frame, label, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            videoWriter.write(frame)
            process_frame_time = time.time()
            print('处理帧fps:',1/(process_frame_time-start_time))
            # cv2.imshow("Frame", frame)
            # cv2.waitKey(1)
        frame_count += 1
    videoCapture.release()
    videoWriter.release()


#TODO
def detect_img():

    pass

if __name__=='__main__':
    # detect_video(r'C:\PythonProject\dataset\black_river\ch06_20191101091604.mp4','C:\PythonProject\jing_vision\classification\keras_tf.h5')
    for name in [r'/Data/jing/black_river/ch06_20191101091604.mp4',
                 '/Data/jing/black_river/ch06_20191113130115.mp4',
                 '/Data/jing/black_river/ch06_20191103080800.mp4',
                 '/Data/jing/black_river/ch06_20191103091815.mp4'
                 '/Data/jing/black_river/ch06_20191113152153.mp4',
                 '/Data/jing/black_river/ch06_20191113163211.mp4',
                 '/Data/jing/black_river/ch06_20191101172812.mp4']:
        detect_video(name,'/home/create/jing/jing_vision/pth/classification/squeeze.pth')



