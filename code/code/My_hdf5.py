import numpy as np
import pandas as pd
import face_recognition
import cv2
import h5py
#读取保存的脸的数量
def read_record_face_number(hdf_dir):
    hdfFile=h5py.File(hdf_dir,'r+')
    unknown_number=np.array(hdfFile['known_people_number'])
    hdfFile.close()
    return unknown_number[0]

def unknown_number_add(hdf_dir,n):
    hdfFile=h5py.File(hdf_dir,'r+')
    hdfFile['unknown_people_number'][()]+=n
    hdfFile.close()
#返回unknown_people_number
def read_unknown_number(hdf_dir):
    hdfFile=h5py.File(hdf_dir,'r+')
    unknown_number=np.array(hdfFile['unknown_people_number'])
    hdfFile.close()
    return unknown_number

#删除一个hdf所有dataset
def delall_from_hdf(hdf_dir):
    hdfFile=h5py.File(hdf_dir,'r+')
    for key in hdfFile.keys():
        hdfFile.__delitem__(key)
    hdfFile.close()

#添加商品信息
def add_in_products(hdf_dir,product_name,product_position):
    hdfFile=h5py.File(hdf_dir,'r+')
    group=hdfFile['product_info']
    position=np.array([product_position['xmin'],product_position['xmax'],product_position['ymin'],product_position['xmax']])
    group.create_dataset(product_name, data=position)
    hdfFile.close()
    
#读取商品的positino
def read_p_position(hdf_dir,product_name):
    hdfFile=h5py.File(hdf_dir,'r+')
    group=hdfFile['product_info']
    result=np.array(group[product_name])
    position=dict()
    position['xmin'],position['xmax'],position['ymin'],position['ymax']=result[0],result[1],result[2],result[3]
    hdfFile.close()
    return position
    
    
#添加一个dataset----->ok
def add_in_hdf(hdf_dir,face_name,face_encoding):
    hdfFile=h5py.File(hdf_dir,'r+')
    uid=hdfFile['known_people_number'][()]
    group1=hdfFile['face_encoding']
    group1.create_dataset('ID_'+str(uid), data=face_encoding)
    group2=hdfFile['Uname']
    group2.create_dataset('ID_'+str(uid), data=face_name)
    hdfFile['known_people_number'][()]+=1
    hdfFile.close()
    
#删除一个dataset--->ok
def del_from_hdf(hdf_dir,UID):
    hdfFile=h5py.File(hdf_dir,'r+')
    group1=hdfFile['face_encoding']
    group2=hdFile['Uname']
    group1.__delitem__(UID)
    group2.__delitem__(UID)
    hdfFile.close()
    
    
#显示hdf所有key
def show_keys_hdf(hdf_dir):
    hdfFile = h5py.File(hdf_dir,'r+')
    print(hdfFile.keys())
    hdfFile.close()

# 输入一个hdf5文件的路径，返回一个known_face_encodings，一个known_face_names
# 将人脸编码与人名对应----->ok
def load_data(hdf_dir): 
    hdfFile=h5py.File('hdf5/datatest.hdf5','r+')
    i=0
    group_ID=hdfFile['Uname']
    group_face=hdfFile['face_encoding']
    for key in group_ID.keys():
        if i==0:
            known_face_encodings=[group_face[key][()]]
            known_face_names=[group_ID[key][()]]
            i+=1
            continue
        known_face_encodings.append(group_face[key][()])
        known_face_names.append(group_ID[key][()])
    hdfFile.close()
    return known_face_encodings, known_face_names

#创建一个hdfFile
def create_hdfFile(hdf_dir):
    hdfFile=h5py.File(hdf_dir,'w')
    hdfFile.create_dataset('unknown_people_number', data=0)# 新建一个表示未知人数的number
    hdfFile.create_dataset('known_people_number', data=0)#
    hdfFile.create_dataset('gaze_number', data=0)#
    hdfFile.create_group('face_encoding')#然后由ID索引一个group 包含所有的人脸信息
    hdfFile.create_group('Uname')#由ID 索引name
    hdfFile.create_group('product_info') #一个group 包含商品信息
    hdfFile.create_group('gaze_records')#一个geze 的全部记录
    hdfFile.close()

def add_gaze_record_in(hdf_dir,dataset):
    hdfFile=h5py.File(hdf_dir,'r+')
    uid=hdfFile['gaze_number'][()]
    group1=hdfFile['gaze_records']
    group1.create_dataset('ID_'+str(uid), data=dataset)
    hdfFile['gaze_number'][()]+=1
    hdfFile.close()
    
def clear_gaze_records(hdf_dir):
    hdfFile=h5py.File(hdf_dir,'r+')
    group1=hdfFile['gaze_records']
    for key in group1.keys():
        group1.__delitem__(key)
    hdfFile['gaze_number'][()]=0
    hdfFile.close()
    
def read_gaze_records(hdf_dir):
    hdfFile=h5py.File('hdf5/datatest.hdf5','r+')
    i=0
    group=hdfFile['gaze_records']
    if len(group)==0:
        return []
    for key in group.keys():
        if i==0:
            gaze_record=[group[key][()]]
            i+=1
            continue
        gaze_record.append(group[key][()])
    hdfFile.close()
    return  gaze_record
