import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import math
import folium
import webbrowser

# pi = 3.1415926535897932384626;
a = 6378245.0
ee = 0.00669342162296594323
pi=np.pi
class PointF:
    def __init__(self,lat,lon):
        self.lat=lat
        self.lon=lon
def transform(lat,lon):
    dLat = transformLat(lon - 105.0, lat - 35.0)
    dLon = transformLon(lon - 105.0, lat - 35.0)
    radLat = lat / 180.0 * pi
    magic = math.sin(radLat)
    magic = 1 - ee * magic * magic
    sqrtMagic = math.sqrt(magic)
    dLat = (dLat * 180.0) / ((a * (1 - ee)) / (magic * sqrtMagic) * pi)
    dLon = (dLon * 180.0) / (a / sqrtMagic * math.cos(radLat) * pi)
    mgLat = lat + dLat
    mgLon = lon + dLon
    pointF=PointF(mgLat,mgLon)
    return pointF
#             if (outOfChina(lat, lon))
#             {
#                 return new pointF(lat, lon);
#             }
def transformLat(x,y):
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y+ 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * pi) + 20.0 * math.sin(2.0 * x * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * pi) + 40.0 * math.sin(y / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * pi) + 320 * math.sin(y * pi / 30.0)) * 2.0 / 3.0
    return ret

def transformLon(x,y):
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1* math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * pi) + 20.0 * math.sin(2.0 * x * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * pi) + 40.0 * math.sin(x / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * pi) + 300.0 * math.sin(x / 30.0* pi)) * 2.0 / 3.0
    return ret

def gcj_To_Gps84(lat,lon): 
    """
    return lat,lon
    """
    gps = transform(lat, lon)
    lontitude = lon * 2 - gps.lon
    latitude = lat * 2 - gps.lat
    #pointF=PointF(latitude,lontitude)
    #return pointF
    return latitude,lontitude

def bd09_To_Gcj02(bd_lat,bd_lon):
    x,y = bd_lon - 0.0065, bd_lat - 0.006
    z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * pi)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * pi)
    gg_lon = z * math.cos(theta)
    gg_lat = z * math.sin(theta)
    pointF=PointF(gg_lat, gg_lon)
    return  pointF

def bd09_To_Gps84(bd_lat, bd_lon):
    gcj02 = bd09_To_Gcj02(bd_lat, bd_lon)
    map84 = gcj_To_Gps84(gcj02.lat,gcj02.lon)
    return map84

def calDist(forward,now):
    dx2=(forward[0]-now[0])**2
    dy2=(forward[1]-now[1])**2
    return math.sqrt(dx2+dy2)

