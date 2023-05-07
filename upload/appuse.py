import streamlit as st
from numpy import floor

from detect_picture_more import yolov5_use
from torchvision import transforms
from PIL import Image
import cv2
from torch.utils.data import Dataset
from detect_picture_more2 import yolov5_use2
from detect_picture_more3 import yolov5_use3
import torch
import numpy as np
import os
import zipfile
import av
import shutil
from streamlit_webrtc import webrtc_streamer
from scipy import misc

def download(upload_file,save_path=r"/home/luofan/kivy-venv/yolov5/mid"):
    if upload_file is not None:
        file_bytes = np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        # 保存图片
        path=os.path.join( save_path,upload_file.name)
        cv2.imwrite(path, opencv_image)
        return path,opencv_image
        
def download_zip(upload_file,save_path=r"/home/luofan/kivy-venv/yolov5/mid2"):
    if upload_file is not None:
        file_name=upload_file.name.split(".")[0]
        path=os.path.join(save_path,file_name)
        with zipfile.ZipFile(upload_file,"r") as z:
           z.extractall(path=path)
        return path
        
def alanye(weights,imgsz,save_txt,nosave,picshow,number,path='.'):
    if(st.button("OK")):
         #st.progress(10)   需要参数
         im0,result,result_soft=yolov5_use2(weights,path,imgsz,save_txt,nosave)
         st.session_state.im0=im0
         st.session_state.result=result
         st.session_state.result_soft=result_soft
         st.session_state.files=os.listdir(path)
         st.session_state.numpic=len(st.session_state.files)
         if(mode=="照片分析"):
               st.session_state.path_all=[]
               st.session_state.opencv_image=[]
               for i in range(0,picture_num):
                   path,opencv_image= download(upload_file[i])
                   st.session_state.path_all.append(path)
         elif(mode=="数据包分析"):
               path = download_zip(upload_file)
         
         
    if(len(st.session_state)>4):
         if(st.session_state.numpic>1):
               number=st.sidebar.slider('选择第几张照片',1,st.session_state.numpic) 
         st.success("分析完成")   
         if(st.session_state.result[number-1]!=0):
          #shutil.rmtree(path)
               if(picshow=='是'):st.image(st.session_state.im0[number-1], channels="BGR")
               for i in range(0,len(st.session_state.result[number-1])):
                   st.write("预测结果是：",classall[st.session_state.result[number-1][i]])
                   st.write("预测概率是：",str(floor(st.session_state.result_soft[number-1][i].detach().numpy()*100)[0]),'%')
         else:
               st.warning("Warning") 
               st.write("没有目标!(noting:可能为拍照不清晰导致)") 
        
class VideoProcessor:
    def recv(self,frame):
      img=frame.to_ndarray(format="bgr24")
      img=cv2.cvtColor(cv2.Canny(img,100,200),cv2.COLOR_GRAY2BGR)
      #save_path=r"/home/luofan/kivy-venv/yolov5/mid4/out.jpg"
      #cv2.imwrite(save_path,img)
      weights='runs/train/exp/weights/last.pt'  # model path or triton URL
      source=r"/home/luofan/kivy-venv/yolov5/mid4"  # file/dir/URL/glob/screen/0(webcam)
      imgsz=(640, 640)
      save_txt=False
      nosave=True
      #im0,tar=yolov5_use3(img,weights,source,imgsz,save_txt,nosave)
      if(tar==0):st.write("选中不是zip文件，请重新选择！")
      else:st.write("含有目标")
      #os.remove(save_path) 
      return av.VideoFrame.from_ndarray(img,format="bgr24")#im0[0]
      
   
   

if __name__ == '__main__':
    weights='runs/train/exp/weights/last.pt'  # model path or triton URL
    source='mid'  # file/dir/URL/glob/screen/0(webcam)
    imgsz=(640, 640)
    save_txt=False
    nosave=True
    classall=['0 ppb','10 ppb','100 ppb','10e3 ppb','10e4 ppb','10e5 ppb']
    st.sidebar.markdown('层析分析(Machine Learning)')
    st.balloons()
    mode=st.sidebar.selectbox("选择模式",["照片分析","数据包分析","实时分析"])
    
    if(mode=="照片分析"):
        upload_file=st.file_uploader("请选择一张图片",help="上传应该为jpg或png格式图片",type=['png', 'jpg'],key="上传总量小于200M",accept_multiple_files=True) #
        picshow=st.sidebar.radio("是否查看处理好的图片",['是','否'])
        picture_num=len(upload_file)
        flag=0
        if upload_file is not None:
           for i in range(0,picture_num):
               if (upload_file[i].name.endswith('.jpg') or upload_file.name[i].endswith('.png'))!=1:
                   flag=1
           if flag==0:
                   #st.session_state.opencv_image.append(opencv_image)
               save_path=r"/home/luofan/kivy-venv/yolov5/mid"
               number=1
               alanye(weights,imgsz,save_txt,nosave,picshow,number,path=save_path)                   
               if(len(st.session_state)>4):
                   for i in range(0,picture_num):
                       os.remove(st.session_state.path_all[i]) 
           else:
               st.write("选中不是图片，请重新选择！")
               
    elif(mode=="数据包分析"):
        #上传数据包则返回txt文件
        #lookway=st.sidebar.radio("选择查看结果的方式",['数值','输入'])
        upload_file=st.file_uploader("请上传zip数据压缩包",help="上传应该为zip格式包",type=['zip'],key="上传总量小于200M")
        picshow=st.sidebar.radio("是否查看处理好的图片",['是','否'])
        txtdown=st.sidebar.radio("是否输出txt文件",['是','否'])
        if upload_file is not None:
           if upload_file.name.endswith('.zip'):# or upload_file.name.endswith('.png'):
               if(txtdown=='是'):
                  save_txt=True
               else:
                  save_txt=False
               number=1
               if(len(st.session_state)>4):
                  if(st.session_state.numpic>1):
                     number=st.sidebar.slider('选择第几张照片',1,st.session_state.numpic) 
               alanye(weights,imgsz,save_txt,nosave,numpic,picshow,number,path=path)   
               if(txtdown=='是' and len(st.session_state)>4):
                  with open('/home/luofan/kivy-venv/yolov5/mid3/mid.txt') as file:
                     st.download_button(label="点击下载",data=file,file_name="分析结果.txt")
               shutil.rmtree(path)
           else:
               st.write("选中不是zip文件，请重新选择！")
               
    elif(mode=="实时分析"):
        st.write("实时监测分析")
        webrtc_streamer(key="example",video_processor_factory=VideoProcessor)
        

# streamlit run appuse.py




