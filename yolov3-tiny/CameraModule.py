from numpy import sin,cos,tan,arctan
import numpy as np
from cv2 import cv2
from glob import glob
import os

class camera:
    def __init__(self):
        if not os.path.exists("./camera/cameraMatrix.txt") or not os.path.exists("./camera/distCoeffs.txt"):
            print("camera files not found")
            return        
        self.cameraMatrix=np.loadtxt("./camera/cameraMatrix.txt",delimiter=',')
        self.distCoeffs=np.loadtxt("./camera/distCoeffs.txt",delimiter=',')
        self.h=np.loadtxt("./camera/height.txt")
        self.x0=self.cameraMatrix[0,2]
        self.y0=self.cameraMatrix[1,2]
        self.fx=self.cameraMatrix[0,0]
        self.fy=self.cameraMatrix[1,1]
        self.f=(self.fx+self.fy)/2
        

    def undistort(self,xy):
        """
        输入待校正点的像素坐标，输出校正后的像素坐标
        """
        xy=xy.reshape(-1,1,2)
        xy_undistort=cv2.undistortPoints(xy,self.cameraMatrix,self.distCoeffs).reshape(-1,2)
        xy_undistort=np.hstack([xy_undistort,np.ones((xy_undistort.shape[0],1))])
        xy_undistort=xy_undistort.dot(self.cameraMatrix)
        xy_undistort[:,0]+=self.x0
        xy_undistort[:,1]+=self.y0
        return xy_undistort[:,:2]
    def get_dist(self,u,v,x,y):
        """
        输入原图坐标，进行校正，输出距离
        u,v:消隐点横、纵坐标（像素）
        x,y:前车轴线与路面交点的坐标（像素）
        h:摄像机安装高度
        """
        #校正 
        """
        uv=self.undistort(np.array([u,v]))
        u=uv[0,0]
        v=uv[0,1]
        xy=self.undistort(np.array([x,y]))
        x=xy[0,0]
        y=xy[0,1]
        """
        #计算距离
        phi=arctan((v-self.y0)/self.fy)
        omega=arctan((self.x0-u)*cos(phi)/self.fx)
        d=self.h/tan(phi+arctan((y-self.y0)/self.f))
        distance_y=d*cos(omega)
        distance_x=cos(omega)*np.sqrt(d**2+self.h**2)*(x-self.x0)*cos(arctan((y-self.y0)/self.f))/self.f
        return distance_x,distance_y
def calib(imgpath,camroot,nx,ny):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)*290
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    images = glob(imgpath+'/*.jpg')
    if len(images) > 0:
        print("images num for calibration : ", len(images))
    else:
        print("No image for calibration.")
        return
    
    ret_count = 0
    for _, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = (img.shape[1], img.shape[0])
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        #cv2.drawChessboardCorners(img,(nx,ny),corners,ret)
 
        # If found, add object points, image points
        if ret == True:
            ret_count += 1
            objpoints.append(objp)
            imgpoints.append(corners)
            print("read image:",ret_count)
    if len(objpoints)==0:
        print("failed to find corners")
        return        
    ret, cameraMatrix, distCoeffs, _, _ = cv2.calibrateCamera(objpoints, imgpoints,img_size, None, None)
    print('Do calibration successfully')
    np.savetxt(camroot+"cameraMatrix.txt",cameraMatrix,fmt="%.18f",delimiter=',')
    np.savetxt(camroot+"distCoeffs.txt",distCoeffs,fmt="%.18f",delimiter=',')
    return ret,cameraMatrix, distCoeffs     
    

if __name__=="__main__":    
    #camroot=r"C:\Users\Lenovo\Desktop\lvmiao\pic\camera\\" #r"d:\pic\\"
    #校正
    #calib(r"C:\Users\Lenovo\Desktop\lvmiao\pic\old\\",camroot,5,8)
    """
    cam=camera()
    img=cv2.imread("d:/distort.jpg")
    img2=cv2.undistort(img,cam.cameraMatrix,cam.distCoeffs)
    cv2.imwrite("d:/undistort.jpg",img2)
    #原图      
    xy=np.array([[282,2619.59],[102,2928.55],[1181,2115.12],[1577,1894.29],[1590,1918.44],[1653,1975.88],  
            [1589,1996.87],[1597,2045.19],[1606,2109.35],[1617,2205.59],[1640,2354.50],[1678,2630.55],[1787,3269.77]]) 
    d2=cam.getdist(1550.08,1708.86,xy[:,1],59.5)
    print(d2)
    """