import numpy as np 
from cv2 import cv2
import CameraModule
import pandas as pd 

#余弦匹配
def cal_cos(a,b):
    a=a.reshape(1,-1)
    b=b.reshape(-1,1)
    dot=a.dot(b)
    dis=np.linalg.norm(a)*np.linalg.norm(b)
    cos=dot/dis
    return cos[0][0] #从二维array里面把数字扒拉出来

class CarClass:
    #类变量-相机参数
    camera=CameraModule.camera()
    surf = cv2.xfeatures2d.SURF_create(400)# 直接查找关键点和描述符
    LaneParam_DEFAULT=np.array([[ 1.62326809e+00, -2.89598008e+01],
       [-9.01447118e-01,  1.48736288e+03],
       [-2.95902362e+00,  2.66325624e+03],
       [ 3.29744098e+00, -8.78764773e+02]])
    VanishingPoint_DEAFAULT=[962,571]
    def __init__ (self,img,top,left,width,height,
                  frame,
                  LaneParam=LaneParam_DEFAULT,
                  vanishing_point=VanishingPoint_DEAFAULT,
                  ):
        #为了画图临时加的frame
        self.frame=frame

        #几何参数
        self.left=left
        self.top=top
        self.width=width
        self.height=height

        #计算距离
        self.dx,self.dy=CarClass.camera.get_dist(vanishing_point[0],vanishing_point[1],self.left+self.width/2,self.top+self.height)
        #纵向距离变化量
        self.delta_distance=999

        #求车辆尾部中心在地面投影的像素坐标（考虑中心投影
        self.x_center=self.get_x_center(vanishing_point)

        self.car_num=-1 #初始化车辆编号        
        
        #计算车道
        #self.lane_num=-1 #所在车道编号
        self.lane_num=self.get_lanenum(LaneParam)
               
        #颜色信息
        self.img=img
        self.hist_H,self.hist_S,self.hist_V=self.get_HSV_hist(img)
    def get_x_center(self,vanishing_point):
        """
        畸变校正、求车辆尾部中心的横坐标
        """
        """
        if self.left+self.width/2>CarClass.camera.x0:#在右边
            #x_center=self.left+self.width-CarClass.camera.f*0.9/self.dy
            #dx为最右端的距离
            self.dx,self.dy=CarClass.camera.get_dist(vanishing_point[0],vanishing_point[1],self.left+self.width,self.top+self.height)
            x_center=(self.dx-0.9)/self.dy*CarClass.camera.f+CarClass.camera.x0
        else:
            #x_center=self.left+CarClass.camera.f*0.9/self.dy  
            #dx为最左端的距离
            self.dx,self.dy=CarClass.camera.get_dist(vanishing_point[0],vanishing_point[1],self.left,self.top+self.height)
            x_center=(self.dx+0.9)/self.dy*CarClass.camera.f+CarClass.camera.x0 
        #重算dx
        self.dx,self.dy=CarClass.camera.get_dist(vanishing_point[0],vanishing_point[1],x_center,self.top+self.height)
        #undist_center=CarClass.camera.undistort([x_center,self.height+self.top])
        #x_center=undist_center[0,0]
        """
        x_center=int(self.left+self.width/2)
        return x_center

    def get_HSV_hist(self,img):
        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        t=cv2.calcHist([hsv],[0], None, [180], [0, 180])
        hist_H=t/np.sum(t)
        t=cv2.calcHist([hsv],[1], None, [256], [0, 255])
        hist_S=t/np.sum(t)
        t=cv2.calcHist([hsv],[2], None, [256], [0, 255])
        hist_V=t/np.sum(t)
        return hist_H,hist_S,hist_V

    def cos_matching(self,another_car):
        cosh=cal_cos(self.hist_H,another_car.hist_H)
        coss=cal_cos(self.hist_S,another_car.hist_S)
        #cosv=cal_cos(self.hist_V,another_car.hist_V)
        return cosh*0.4+coss*0.6#+cosv*0.1
    def set_car_num(self,num):
        self.car_num=num
    def set_delta_distance(self,previous_distance):
        self.delta_distance=self.dy-previous_distance
    def position_matching(self,another_car):
        dist=(self.x_center-another_car.x_center)**2+(self.top+self.height-another_car.top-another_car.height)**2+0.0000001
        return 1/dist
    def set_res_size(self,previous_car):
        self.width*self.height/(previous_car.width*previous_car.height)
    def size_matching(self,another_car): #小尺寸除以大尺寸。越大（接近1）越好
        if another_car.width>self.width:
            return (self.width/another_car.width+self.height/another_car.height)/2
        else:
            return (another_car.width/self.width+another_car.height/self.height)/2

    def get_lanenum(self,LaneParam):
        """
        输入待定点的横纵坐标和车道线的系数（车道线方程：y=b1*x+b0），返回待定点所在车道
        """
        '''
        if np.abs(self.dx)<=2:
            return 1
        elif np.abs(self.dx)<=4:
            return 2
        elif np.abs(self.dx)<=6:
            return 3
        else:
            return 4
        '''
        y=self.top+self.height
        lanes=[]
        n=LaneParam.shape[0]
        
        if n<2:
            return -1
        for i in range(n):
            #把线画出来
            f=lambda y:LaneParam[i,0]*y+LaneParam[i,1]
            cv2.line(self.frame,(int(f(400)),400),(int(f(900)),900),(0,0,255),thickness=3)
            lanes.append(LaneParam[i,0]*y+LaneParam[i,1])
        lanes=np.sort(lanes) #排序 从小到大
        for i in range(n-1):
            if self.x_center<lanes[i+1]:
                return i+1 #从1开始 最左一个车道为1
        else:
            return n
        

    def SURFmatch(self,previous_car):
        #https://www.cnblogs.com/Lin-Yi/p/9435824.html
        kp1, des1 = CarClass.surf.detectAndCompute(self.img,None)
        kp2, des2 = CarClass.surf.detectAndCompute(previous_car.img,None)
        # kdtree建立索引方式的常量参数
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50) # checks指定索引树要被遍历的次数
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        # 进行匹配搜索
        matches = flann.knnMatch(des1, des2, k=2)
        # 寻找距离近的放入good列表
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        return len(good)


def get_lane_param(imgname):
    """
    返回消失点array([x,y])、车道线系数矩阵[b1,b0]  x=b1*y+b0
    """
    df=pd.read_csv(r"C:\Users\Lenovo\Desktop\lvmiao\py\LaneParameters.txt",delimiter='\t')
    LaneParam=df[df.name==imgname][["b1","b0"]].values
    n=LaneParam.shape[0]
    DEFAULT_y=600 #600:消失点纵坐标经验值
    ######################取数目正确的车道线################
    if n<3: #测试。车道检测不完全时用上一次的结果
        return np.array([-1,-1]),LaneParam  #返回异常值
    #斜率全正或全负
    if np.abs(np.sum(np.sign(LaneParam[:,0])))==n:
        return np.array([-1,-1]),LaneParam  #返回异常值
    B=np.hstack([-np.ones((n,1)),LaneParam[:,0].reshape(-1,1)])
    l=-1*LaneParam[:,1].reshape(-1,1)
    BT=np.transpose(B)
    x=np.dot(np.dot(np.linalg.inv(np.dot(BT,B)),BT),l)
    V=np.dot(B,x)-l
    #求出稳定的消失点
    if V.transpose().dot(V)[0,0]/n<=2000 and np.abs(x[1,0]-DEFAULT_y)<30: 
        return x.flatten(),LaneParam #x:np.array([消失点横坐标,消失点纵坐标])   
    else:
        return np.array([-1,-1]),LaneParam  #返回异常值
    ######################取一对正确的车道线################
    """
    if n<2:#不能计算消失点
        return np.array([-1,-1]),LaneParam  #返回异常值
    B=np.hstack([-np.ones((n,1)),LaneParam[:,0].reshape(-1,1)])
    l=-1*LaneParam[:,1].reshape(-1,1)
    BT=np.transpose(B)
    x=np.dot(np.dot(np.linalg.inv(np.dot(BT,B)),BT),l)
    V=np.dot(B,x)-l
    #求出稳定的消失点
    if V.transpose().dot(V)[0,0]/n<=2000 and np.abs(x[1,0]-DEFAULT_y)<30: 
        return x.flatten(),LaneParam #x:np.array([消失点横坐标,消失点纵坐标])   
    else:
        #找最接近经验值的一组
        min_delta_y=1000
        for i in range(n-1):
            for j in range(i+1,n):
                vp_y=(LaneParam[i,1]-LaneParam[j,1])/(LaneParam[j,0]-LaneParam[i,0])
                if np.abs(vp_y-DEFAULT_y)<min_delta_y:
                    mini=i
                    minj=j
                    min_delta_y=np.abs(vp_y-DEFAULT_y)
        if min_delta_y<20: #有符合要求的车道线对i,j。考虑内插？
            #重新用符合要求的车道计算消失点
            vp_y=(LaneParam[mini,1]-LaneParam[minj,1])/(LaneParam[minj,0]-LaneParam[mini,0])
            vp_x=LaneParam[mini,0]*vp_y+LaneParam[mini,1]
            LaneParam=np.vstack([LaneParam[mini],LaneParam[minj]]) #更新。只取有效车道           
            return np.array([vp_x,vp_y]),LaneParam        
        else: #没有符合要求的车道线对i,j
            return np.array([-1,-1]),LaneParam #返回异常值
    """
    """
    ##################取靠近经验值的所有车道线#######################
    if n<2:#不能计算消失点
        return np.array([-1,-1]),LaneParam  #返回异常值
    B=np.hstack([-np.ones((n,1)),LaneParam[:,0].reshape(-1,1)])
    l=-1*LaneParam[:,1].reshape(-1,1)
    BT=np.transpose(B)
    x=np.dot(np.dot(np.linalg.inv(np.dot(BT,B)),BT),l)
    V=np.dot(B,x)-l
    #求出稳定的消失点
    if V.transpose().dot(V)[0,0]/n<=2000 and np.abs(x[1,0]-DEFAULT_y)<30: 
        return x.flatten(),LaneParam #x:np.array([消失点横坐标,消失点纵坐标])  
    else: #残差大，两两求解， #满足dy在经验值上下20px以内的都留下
        valid_list=[]      
        for i in range(n-1):
            for j in range(i+1,n):
                vp_y=(LaneParam[i,1]-LaneParam[j,1])/(LaneParam[j,0]-LaneParam[i,0])
                if np.abs(vp_y-DEFAULT_y)<30:
                    if i not in valid_list:
                        valid_list.append(i)
                    if j not in valid_list:
                        valid_list.append(j)
        LaneParam=LaneParam[valid_list]
        #重新用符合要求的车道计算消失点
        n=LaneParam.shape[0]
        if n<2:
            return np.array([-1,-1]),LaneParam  #返回异常值
        B=np.hstack([-np.ones((n,1)),LaneParam[:,0].reshape(-1,1)])
        l=-1*LaneParam[:,1].reshape(-1,1)
        BT=np.transpose(B)
        x=np.dot(np.dot(np.linalg.inv(np.dot(BT,B)),BT),l)
        V=np.dot(B,x)-l          
        return x.flatten(),LaneParam
    """    


if __name__=="__main__":
    #filename=r"D:\BaiduNetdiskDownload\20190429\pic\bos_5b5c4993fd443569_1556428189000.jpg"
    #src=cv2.imread(filename)
    pass