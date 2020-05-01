import pymssql
import numpy as np
import pandas as pd
import coordinate
from PIL import Image
from sklearn.externals import joblib
import copy
import time
from cv2 import cv2

class Road:
    def __init__(self, rid,rname, snodeid,enodeid,rtype):
        """
        道路id，名称，起点编号，终点编号，道路类型
        """
        self.rid = rid
        self.rname = rname
        self.snodeid = snodeid
        self.enodeid = enodeid
        self.rtype = rtype
class GPSinfo:
    def __init__(self,lon,lat,pang,jpgpath):
        self.lon=lon
        self.lat=lat
        self.pang=pang
        self.jpgpath=jpgpath

def Image_Process(jpgpath):
    """
    输入用于验证的图片的绝对地址，以列表形式
    返回处理后的数据：array列表
    """
    #hists=[]
    #获取图像
    img = Image.open(jpgpath).convert('LA')
   
    #缩小图像
    x, y = img.size
    img=img.resize((round(x/10),round(y/10)))
    #切割图像
    im=np.array(img)[:,:,0]
    im=im[0:round(1/2*im.shape[0]),:].reshape(1,-1)
    #hists.append(copy.deepcopy(im/max(im)))
    """
    hist=np.zeros(256)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            hist[im[i,j]]+=1
    hists.append(copy.deepcopy(hist/max(hist)))
    return hists
    """
    return im

def Image_Process2(img):
    """
    输入用于验证的图片的数组，以列表形式
    返回处理后的数据：array列表
    """
    #缩小图像
    x, y ,_= img.shape
    img=cv2.resize(img,(192,108))
    #切割图像
    im=np.array(img)[:,:,0]
    im=im[0:round(1/2*im.shape[0]),:].reshape(1,-1)
    #hists.append(copy.deepcopy(im/max(im)))
    return im

def get_roadtype(jpgpath,lastrtype,img,svc):
    """
    if i<140 or i>=1231:
        return 0
    else:
        return 1
    """
    try:
        imghist=Image_Process2(img) 
        #imghist=Image_Process(jpgpath) 
        rtype=svc.predict(imghist)[0]   
        #print(rtype) 
    except Exception as e: #出错，返回上一点的道路类型
        print(jpgpath+'fail to classify') 
        print(str(e))
        return lastrtype       
    if rtype==1: #高架上
        return 1
    elif rtype==0: #高架下 
        return 0
    else: #隧道
        return lastrtype

#原版
'''
def match(lon,lat,pang,lastrid,lastsnode,lastenode,a,conn,i):
    """
    #lon&lat:当前点的经纬度
    #pang:当前车的方位角（没用上）
    #lastsnode&lastenode:上一个路段的snodeid、enodeid（两个端点编号）
    #a:权重 a[0]:距离 a[1]:邻接 a[2]:道路类型
    """
    threashold=0.0005 #搜索半径。单位°，约等于50米
    cmdStr=f"select id,snodeid,enodeid,rang,geometry::STGeomFromText('Point({lon} {lat})',0).STDistance(Shape) as dist,rtype, \
        shape.STAsText() as wkt,kind \
    from dbo.shroads where geometry::STGeomFromText('Point({lon} {lat})',0).STDistance(Shape)<{threashold}"
    #x是车方位角和路段方向之差。取锐角。（可能有点问题
    #ang=lambda x:min(x,abs(x-180),360-x)
    #判断是否与上一个路段邻接 判断标准：当前路段和上一路段，snodeid、enodeid排列组合有相等的，说明共享一个结点。邻接的取0，不邻接取1
    adj=lambda x:min(abs(x['snodeid']-lastsnode),abs(x['snodeid']-lastenode),abs(x['enodeid']-lastsnode),abs(x['enodeid']-lastenode),1)
    #adj=lambda x:min(abs(x['snodeid']-lastenode),abs(x['enodeid']-lastsnode),1)
    df=pd.read_sql(cmdStr,conn) #获取候选路段。试图用dataframe广播减少循环
    #df=df.apply(pd.to_numeric,errors='ignore')
    try:
        #判断是否到高架附近
        if sum(df['rtype']==1)>0:   #到高架附近                    
            rt_true=get_roadtype(i) #由视觉判断是否高架，是高架返回1，不是高架返回0。
            #加权取最小的一个路段的id。距离归一化、邻近、道路类型
            score_r=rightside(pang,df)
            #print(df['id'].values)
            #print(score_r)
            score_ramp=is_ramp(df)
            score=df['dist']/df['dist'].sum()*a[0]+df.apply(adj,axis=1)*a[1]+abs(rt_true-df['rtype'])*a[2]+a[3]*score_r/sum(score_r)+a[4]*score_ramp
            ind=score.idxmin()              
        else: #不在高架附近，少一个判断道路类型              
            score=df['dist']/df['dist'].sum()*a[0]+df.apply(adj,axis=1)*a[1]+a[3]* rightside(pang,df)
            ind=score.idxmin()
            
        #ind=(df['dist']/df['dist'].sum()*0.9+df.apply(adj,axis=1)*0.1).idxmin() # +abs(df['rang']-pang).apply(ang)*0.2
        #ind=(df['dist']/df['dist'].sum()*0.8+df.apply(adj,axis=1)*0.2).idxmin() # 0.61341
        #ind=(df['dist']/df['dist'].sum()*0.8+df.apply(adj,axis=1)*0.1+abs(df['rang']-pang).apply(ang)*0.1).idxmin() #0.973642
        #ind=(df['dist']/df['dist'].sum()*a[0]+df.apply(adj,axis=1)*a[1]+(1-df['rtype'])*a[2]).idxmin() #0.613418
    except Exception as e: #200半径没搜到道路
        try:
            print(str(e))
            threashold=0.003 #试试300
            df=pd.read_sql(cmdStr,conn)
            #df=df.apply(pd.to_numeric,errors='ignore')            
            #判断是否到高架附近
            if sum(df['rtype']==1)>0:   #到高架附近                    
                rt_true=get_roadtype(i) #由视觉判断是否高架，是高架返回1，不是高架返回0。
                #加权取最小的一个路段的id。距离归一化、邻近、道路类型
                score_r=rightside(pang,df)
                score_ramp=is_ramp(df)
                #print(df.iloc[score_r.idxmin(),0])
                score=df['dist']/df['dist'].sum()*a[0]+df.apply(adj,axis=1)*a[1]+abs(rt_true-df['rtype'])*a[2]+a[3]*score_r/sum(score_r)+a[4]*score_ramp
                ind=score.idxmin()              
            else: #不在高架附近，少一个判断道路类型              
                score=df['dist']/df['dist'].sum()*a[0]+df.apply(adj,axis=1)*a[1]+a[3]* rightside(pang,df)
                ind=score.idxmin()
        except:#300m半径内还是没有,返回前一个路段的数值
            return lastrid,lastsnode,lastenode 
    return df.iloc[ind,0],df.iloc[ind,1],df.iloc[ind,2]#返回rid，snodeid，enodeid,distmin
'''

#改版2
def match(gps,lastroad,a,conn,img,svc):
    '''
    #lon&lat:当前点的经纬度
    #pang:当前车的方位角（没用上）
    #lastsnode&lastenode:上一个路段的snodeid、enodeid（两个端点编号）
    #a:权重 a[0]:距离 a[1]:邻接 a[2]:道路类型
    '''
    thisroad=lastroad
    for threashold in [0.0002,0.0005,0.002]:
        cmdStr=f"select id,pyname as name,snodeid,enodeid,rang,geometry::STGeomFromText('Point({gps.lon} {gps.lat})',0).STDistance(Shape) as dist,rtype, \
        shape.STAsText() as wkt,kind \
        from dbo.shroads where geometry::STGeomFromText('Point({gps.lon} {gps.lat})',0).STDistance(Shape)<{threashold}"
        #x是车方位角和路段方向之差。取锐角。（可能有点问题
        #判断是否与上一个路段邻接 判断标准：当前路段和上一路段，snodeid、enodeid排列组合有相等的，说明共享一个结点。邻接的取0，不邻接取1
        adj=lambda x:min(abs(x['snodeid']-lastroad.snodeid),abs(x['snodeid']-lastroad.enodeid),abs(x['enodeid']-lastroad.snodeid),abs(x['enodeid']-lastroad.enodeid),1)
        try:
            df=pd.read_sql(cmdStr,conn) #获取候选路段。试图用dataframe广播减少循环
            if df.shape[0]==0:
                print(threashold,"fail")
                if threashold==0.002:#最大范围也没有找到道路，重新初始化
                    print("not found")
                    thisroad.rid=0
                    thisroad.rname=""
                    thisroad.snodeid=0
                    thisroad.enodeid=0
                    thisroad.rtype=0
                continue
            #判断是否到高架附近                          
            #加权取最小的一个路段的id。距离归一化、邻近、道路类型
            score_r,score_topo=rightside(gps.pang,df)
            #print(df['id'].values)
            #print(score_r)           
            score=df['dist']/df['dist'].sum()*a[0]+df.apply(adj,axis=1)*a[1]*score_topo+a[3]*score_r/sum(score_r)
            if threashold==0.0002 and sum(df['rtype']==1)>0:   #到高架附近 只有最近距离判断（普通道路和高架上无法区别）              
                thisroad.rtype=get_roadtype(gps.jpgpath,lastroad.rtype,img,svc) #由视觉判断是否高架，是高架返回1，不是高架返回0。
                score_ramp=is_ramp(df) #匝道
                score+=abs(thisroad.rtype-df['rtype'])*a[2]+a[4]*score_ramp
            ind=score.idxmin()
            thisroad.rid=df['id'].iloc[ind]
            if df['name'].iloc[ind] is np.nan:
                thisroad.rname=lastroad.rname
            else:
                thisroad.rname=df['name'].iloc[ind]
            thisroad.snodeid=df['snodeid'].iloc[ind]
            thisroad.enodeid=df['enodeid'].iloc[ind]
            break
        except Exception as e: #当前半径没搜到道路
            print(str(e))
            
    return thisroad


def map_matching(a,conn,readpath,writepath,jpgdir):
    """
    #a:权重列表
    #readpath:读按时间排好序的、坐标转换了的轨迹-图片对应文件
    #writepath:记录求得的路段id
    #userdf:uid(车辆编号),lon,lat,pang(车辆方向角),relation(轨迹图片匹配),datetime2,datetime1,trueroad(手动标记的正确路段)
    """
    #userdf=pd.read_csv(r"D:\BaiduNetdiskDownload\20190429样例数据\orderbytimeasc.csv")
    userdf=pd.read_csv(readpath)
    lastroad=Road(0,"",0,0,0)
    #f=open(r"D:\BaiduNetdiskDownload\20190429样例数据\record.csv","w")
    with open(writepath,"w") as f:
        f.write('rid,lon,lat\n')
        for i in range(len(userdf)):
            lat,lon=coordinate.gcj_To_Gps84(userdf['lat'].iloc[i],userdf['lon'].iloc[i]) 
            pang=userdf['pang'].iloc[i]
            jpgpath=jpgdir+userdf['relation'].iloc[i]
            gps=GPSinfo(lon,lat,pang,jpgpath)           
            #print(f"第{i}个点：")
            #lastroad=match(gps,lastroad,a,conn)
            lastroad=match(gps,lastroad,a,conn,1,svc)
            f.write(f"{lastroad.rid},{lon},{lat}\n")
            print(lon,lat)
    print("done")
    print("权重:",a)

def get_gpsinfo(imgname):
    df=pd.read_csv(r"D:\BaiduNetdiskDownload\20190429\orderbytimeasc4.csv")
    info=df[df.relation==imgname][["lon","lat","pang"]].values[0] #有些图片对应多条gps
    return info.flatten()

def Imap_matching(imgdir,imgname,img,a,conn,svc):
    """
    #a:权重列表
    #readpath:读按时间排好序的、坐标转换了的轨迹-图片对应文件
    #userdf:uid(车辆编号),lon,lat,pang(车辆方向角),relation(轨迹图片匹配),datetime2,datetime1,trueroad(手动标记的正确路段)
    """
    info=get_gpsinfo(imgname)
    lon,lat,pang=info
    lastroad=Road(0,"",0,0,0)
    #坐标转换
    lat,lon=coordinate.gcj_To_Gps84(lat,lon) 
    imgpath=imgdir+"\\"+imgname
    gps=GPSinfo(lon,lat,pang,imgpath)           
    lastroad=match(gps,lastroad,a,conn,img,svc)
    return lastroad.rid,lastroad.rname

def score_label(labelpath,truepath):
    #labelpath=r"D:\BaiduNetdiskDownload\20190429样例数据\record.csv"
    #truepath=r"D:\BaiduNetdiskDownload\20190429样例数据\orderbytimeasc.csv"
    lb=pd.read_csv(labelpath)#.iloc[:1252,:] #1252后面的200个点路网数据里没有，没做异常处理，怎么算都肯定不对干脆截掉了
    tr=pd.read_csv(truepath)#.iloc[:1252,:]
    lb['pang']=tr['pang']
    lb['relation']=tr['relation']
    lb['trueroad']=tr['trueroad']
    mismatch=lb[lb['trueroad']!=lb['rid']]
    print("错误率",mismatch.shape[0]/lb.shape[0])
    #print("标记id","真实id")
    #print(mismatch.iloc[:,.iloc[:,[0,8]])
    mismatch.to_csv(r"D:\BaiduNetdiskDownload\20190429\mismatch.csv",index=False)
    return mismatch.shape[0]/lb.shape[0]

def get_nodes(df):
    n=df.shape[0]
    st=np.zeros((n,2))
    ed=np.zeros((n,2))   
    for i in range(df.shape[0]): 
        points=df['wkt'].iloc[i][12:-1].split(',')       
        st[i]=list(map(float,points[0].split(' ')))
        ed[i]=list(map(float,points[-1].split(' ')[1:]))
    return st,ed
def rightside(pang,df):
    st,ed=get_nodes(df)
    num=ed.shape[0]
    rights=np.zeros(num)
    topo=np.zeros(num)
    #从高斯坐标系方位角转笛卡尔方位角
    pang=(450-pang) % 360
    #角度转弧度
    pang=pang/180*np.pi
    #换到与pang夹角为锐角的方向
    for i in range(num):
        #求道路方位角 arctan2值域：-180~180
        rang=np.arctan2(ed[i,1]-st[i,1],ed[i,0]-st[i,0])
        if rang<0: #(-180~0)
            rang+=2*np.pi
        #求和车辆方向夹角
        da=rang-pang
        #判断夹角锐角还是钝角
        if np.cos(da)<0:#钝角，取反方向
            st[i,0],ed[i,0]=ed[i,0],st[i,0]
            st[i,1],ed[i,1]=ed[i,1],st[i,1]
            #print("反向:",df.iloc[i,0],st[i],ed[i])
        #大夹角惩罚：
        rights[i]=np.exp2(-(np.cos(da))**2*180)*num/2
        #大夹角加强拓扑：
        topo[i]=np.exp2(-(np.cos(da))**2)+1
    #有几个道路终点在road[i]右边
    for i in range(0,num):
        for j in range(0,num):
            if i==j:
                continue
            ai=np.arctan2(ed[i,1]-st[i,1],ed[i,0]-st[i,0])
            if ai<0:
                ai+=np.pi*2
            #aj=np.arctan2(ed[j,1]-st[i,1],ed[j,0]-st[i,0]) #终点
            aj=np.arctan2(ed[j,1]+st[j,1]-st[i,1]*2,ed[j,0]+st[j,0]-st[i,0]*2) #中点
            #aj=np.arctan2(ed[j,1]+st[j,1]-ed[i,1]-st[i,1], ed[j,0]+st[j,0]-ed[i,0]-st[i,0])
            if aj<0:
                aj+=np.pi*2
            if ai>aj:
                rights[i]+=1
    #print(topo)
    return rights/sum(rights),topo

def is_ramp(df):
    ramp=np.zeros(df.shape[0])
    for i in range(df.shape[0]):
        kinds=df['kind'].iloc[i].split('|')
        for kind in kinds:
            if kind=='010b':
                ramp[i]=1
                break
    return ramp

def search_param():
    #距离 结点 高架 靠右(小夹角、反馈) 匝道 
    with open (root+"search.csv","a") as f:
        #[2,3,5,2,1]  0.0855
        a=np.random.random((5,))
        map_matching(a,conn,root+"orderbytimeasc4.csv", root+"record.csv",root+"pic\\")
        score=score_label(root+"record.csv",root+"orderbytimeasc4.csv")
        msg=f"错误率：{score}，参数：{a}"
        f.write(msg)
        print(msg)
if __name__=="__main__":
    conn=pymssql.connect(server='127.0.0.1',database='SPDB')
    root="D:\\BaiduNetdiskDownload\\20190429\\"
    svc=joblib.load(r"C:\Users\Lenovo\Desktop\lvmiao\py\gaojia\all_svc_train_model.m")
    sttime=time.time()
    map_matching([2,3,5,2,1],conn,root+"orderbytimeasc4.csv", root+"record.csv",root+"pic\\under\\")
    elapsed=time.time()-sttime
    
    score=score_label(root+"record.csv",root+"orderbytimeasc4.csv")
    print(elapsed)
