#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import random as rand
import time
import numpy as np
import pickle
import sys
import math
from functools import reduce
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps, ImageFile
from delaunay import delaunay
from voronoi import createVoronoiFromDelaunay
import random
#
# Add a prefix to a path-specified filename;
# prefix goes on the filename portion.
#
def addFilenamePrefix( prefix, filename ):
    return os.path.join( os.path.dirname( filename ), prefix + os.path.basename( filename ) )

def generateRandomPoints(count, sizeX, sizeY):
    points = []
    start = time.perf_counter()
    for i in range(count):
        p = (rand.randint(0,sizeX),rand.randint(0,sizeY))
        if not p in points:
            points.append(p)
    print("Punkte generieren: %.2fs" % (time.perf_counter()-start))
    return points

def generateWeightedRandomPoints(count, sizeX, sizeY):
    points = []
    start = time.perf_counter()
    for i in range(count):
        x = rand.randint(0,sizeX/2)-rand.randint(0,sizeX/2) + sizeX/2
        y = rand.randint(0,sizeY/2)-rand.randint(0,sizeY/2) + sizeY/2
        p = (x, y)
        if not p in points:
            points.append(p)
    print ("Punkte generieren: %.2fs" % (time.perf_counter()-start))
    return points

def drawPoints(points, filename, sizeX, sizeY):
    im = Image.new('RGB', (sizeX*10, sizeY*10))
    draw = ImageDraw.Draw(im)
    for p in points:
        px = p[0]*10
        py = p[1]*10
        draw.arc((px, py, px+20,py+20),0,360,fill='white')
    im.save(filename, "JPG")

def drawTriangulation(triangles, filename, sizeX, sizeY, multiplier):
    im = Image.new('RGB', (sizeX*multiplier, sizeY*multiplier))
    draw = ImageDraw.Draw(im)
    start = time.perf_counter()
    for t in triangles:
        r = rand.randint(0,255)
        g = rand.randint(0,255)
        b = rand.randint(0,255)
        p0 = tuple(map(lambda x:x*multiplier, t[0]))
        p1 = tuple(map(lambda x:x*multiplier, t[1]))
        p2 = tuple(map(lambda x:x*multiplier, t[2]))
        drawT = (p0, p1, p2)
        draw.polygon(drawT, fill=(r,g,b,255))
    im.save(filename, "JPEG")
    print("Dreiecke zeichnen: %.2fs" % (time.perf_counter()-start))

def getCenterPoint(t):
    return ((t[0][0]+t[1][0]+t[2][0])/3, (t[0][1]+t[1][1]+t[2][1])/3)

def getTriangleColor(t, im):

    # 3x der Wert in der Mitte + jew. die Ecke / 6.
    color = []
    for i in range(3):
        p = t[i]
        if p[0] >= im.size[0] or p[0] < 0 or p[1] >= im.size[1] or p[1] < 0:
            continue
        color.append(im.getpixel(p))

    p = getCenterPoint(t)
    if p[0] < im.size[0] and p[0] >= 0 and p[1] < im.size[1] and p[1] >= 0:
        centerPixel = im.getpixel(p)
        color = color + [centerPixel]*3

    div = float(len(color))
    color = reduce(lambda rec, x : ((rec[0]+x[0])/div, (rec[1]+x[1])/div, (rec[2]+x[2])/div), color, (0,0,0))
    color = map(lambda x : int(x), color)
    return color

# def getPolygonColor(pol, im):

#     centerPoint = (0,0)
#     color = []
#     count = 0
#     #print ""

#     for p in pol:
#         if p[0] >= im.size[0] or p[0] < 0 or p[1] >= im.size[1] or p[1] < 0:
#             continue
#         count += 1
#         color.append(im.getpixel(p))
#         centerPoint = (centerPoint[0]+p[0], centerPoint[1]+p[1])

#     centerPoint = (centerPoint[0]/count, centerPoint[1]/count)

#     color.append(im.getpixel(centerPoint))
#     color.append(im.getpixel(centerPoint))
#     color.append(im.getpixel(centerPoint))


#     div = float(len(color))
#     color = reduce(lambda rec, x : ((rec[0]+x[0]), (rec[1]+x[1]), (rec[2]+x[2])), color, (0,0,0))
#     color = (color[0]/div, color[1]/div, color[2]/div)
#     # Diese Zeile ergibt KEINEN Sinn!!!!!  Aber anders hab ichs nicht zum Laufen gebracht. Irgendein Fehler mit der Farbe...
#     color = (color[0]/4.0, color[1]/4.0, color[2]/4.0)
#     color = map(lambda x : int(x), color)
#     return color
def getPolygonColor(pol, im):
    # 计算多边形的边界框
    min_x = min(p[0] for p in pol)
    max_x = max(p[0] for p in pol)
    min_y = min(p[1] for p in pol)
    max_y = max(p[1] for p in pol)

    width, height = im.size  # 获取图像宽高
    black_count = 0
    white_count = 0

    # 对边界框中的所有像素点进行检查
    for yy in range(int(min_y), int(max_y)+1):
        # 如果yy不在图像垂直范围内，跳过此行
        if yy < 0 or yy >= height:
            continue
        for xx in range(int(min_x), int(max_x)+1):
            # 如果xx不在图像水平范围内，跳过此列
            if xx < 0 or xx >= width:
                continue
            # 确认点在多边形内
            if point_in_polygon(xx, yy, pol):
                pixel = im.getpixel((xx, yy))
                # 假设原图为黑白（0表示黑，255表示白，或(R,G,B)都相同）
                if isinstance(pixel, tuple):
                    # RGB模式
                    val = pixel[0] # R分量即可，因为R=G=B
                else:
                    # 灰度模式
                    val = pixel
                
                if val < 128:
                    black_count += 1
                else:
                    white_count += 1

    # 如果整个区域没有有效像素(black_count+white_count=0)，可以决定一个默认颜色
    if (black_count + white_count) == 0:
        return (255, 255, 255)  # 或者(0,0,0)，视需要而定

    # 决定填充颜色
    if black_count > white_count:
        return (0, 0, 0)   # 黑色
    else:
        return (255, 255, 255) # 白色


def point_in_polygon(x, y, polygon):
    # 简化版的点在多边形内判断，使用射线法
    num = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(num+1):
        p2x, p2y = polygon[i % num]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def brightenImage(im, value):
    enhancer = ImageEnhance.Brightness(im)
    im = enhancer.enhance(value)
    return im

def drawImageColoredTriangles(triangles, filename, origIm, multiplier):
    (sizeX, sizeY) = origIm.size
    im = Image.new('RGB', (sizeX*multiplier, sizeY*multiplier))
    draw = ImageDraw.Draw(im)
    start = time.perf_counter()
    for t in triangles:
        (r,g,b) = getTriangleColor(t, origIm)
        p0 = tuple(map(lambda x:x*multiplier, t[0]))
        p1 = tuple(map(lambda x:x*multiplier, t[1]))
        p2 = tuple(map(lambda x:x*multiplier, t[2]))
        drawT = (p0, p1, p2)
        draw.polygon(drawT, fill=(r,g,b,255))
    im = brightenImage(im, 3.0)
    ImageFile.MAXBLOCK = im.size[0] * im.size[1]
    im.save(filename, "JPEG", quality=100, optimize=True, progressive=True)    
    

def drawImageColoredVoronoi(polygons, filename, origIm, multiplier=1):
    start = time.perf_counter()
    (sizeX, sizeY) = origIm.size
    im = Image.new('RGB', (sizeX*multiplier, sizeY*multiplier))
    draw = ImageDraw.Draw(im)
    for pol in polygons:
        if len(pol) < 2:
            continue
        (r,g,b) = getPolygonColor(pol, origIm)
        # newPol = map(lambda x: (x[0] * multiplier, x[1]*multiplier), pol)
        # draw.polygon(newPol, fill=(r,g,b,255))
        # newPol = list(map(lambda x: (x[0] * multiplier, x[1] * multiplier), pol))
        newPol = list(pol)
        draw.polygon(newPol, fill=(r, g, b, 255))
    im = brightenImage(im, 3.0)
    ImageFile.MAXBLOCK = im.size[0] * im.size[1]
    im.save(filename, "JPEG", quality=100, optimize=True, progressive=True)
    print("Voronoi zeichnen: %.2fs" % (time.perf_counter()-start))
   
def generateTriangles(points):
    start = time.perf_counter()
    triangles = delaunay(points)
    print("Delaunay-Triangulierung: %.2fs" % (time.perf_counter()-start))
    return triangles

# Der Faktor, der die Anzahl generierter Punkte bestimmt ist der Exponent von v.
# Auf ein Bild der Auflösung 1000x750:
# 1.0 ~ 80   Punkte
# 1.5 ~ 500  Punkte
# 2.0 ~ 3000 Punkte
# 2.2 ~ 9500 Punkte
def findPointsFromImage(im, factor):
    start = time.perf_counter()
    pix = np.array(im)
    points = []

    for row in range(len(pix)):
        for col in range(len(pix[row])):

            v = pix[row][col]
            v = v**float(factor) / float(2**18)
            if np.random.random() < v:
                points.append((col, row))

    print("Anzahl erzeugter Punkte:", len(points))
    print("Punkte extrahieren: %.2fs" % (time.perf_counter()-start))
    return points

def loadAndFilterImage(name):
    start = time.perf_counter()
    orig = Image.open(name)
    im = orig.convert("L")
    im = im.filter(ImageFilter.GaussianBlur(radius=5))
    im = im.filter(ImageFilter.FIND_EDGES)

    im = brightenImage(im, 20.0)

    im = im.filter(ImageFilter.GaussianBlur(radius=5))
    print("Bild laden: %.2fs" % (time.perf_counter()-start))
    return (orig, im)

def tupleToString(t):
    return "{" + str(t[0]) + ", " + str(t[1]) + ", " + str(t[0]) + "}"

def printTriangleList(l):
    for t in l:
        if t != None:
            print(tupleToString(t)),
    print(""),

def removeUnusedLinks(triangles):
    newList = []
    for t in triangles:
        newList[:0] = (t[0],t[1],t[2])
    return newList

def pointsToTriangles(points):
    triangles = []
    for i in range(len(points)-2):
        t = (points[i],points[i+1],points[i+2])
        triangles.append(t)
    return triangles

def readTriangleListFromFile(filename):
    with open(filename, 'r') as f:
        points = pickle.load(f)
    triangles = pointsToTriangles(points)
    return triangles

def saveTriangleListToFile(triangles, filename):

    triangles = removeUnusedLinks(triangles)
    with open(filename, 'w') as f:
        pickle.dump(triangles, f)

def autocontrastImage(input_filename, output_filename):
    start = time.perf_counter()
    im = Image.open(input_filename)
    im = ImageOps.autocontrast(im)
    im.save( addFilenamePrefix( "autocontrasted_", output_filename ), "JPEG" )
    print("Autocontrast Image: %.2fs" % (time.perf_counter()-start)),

def equalizeImage(filename):
    start = time.perf_counter()
    im = Image.open(filename)
    im = ImageOps.equalize(im)
    im.save( addFilenamePrefix( "equalized_", filename ), "JPEG" )
    print("Equalize Image: %.2fs" % (time.perf_counter()-start))

def resizeImage(filename, longestSide, outDirectory="."):
    im = Image.open(filename)
    (width, height) = im.size
    ratioX = float(longestSide) / width
    ratioY = float(longestSide) / height
    ratio = min(ratioX, ratioY)
    im.thumbnail((width*ratio, height*ratio), Image.ANTIALIAS)
    newFilename = os.path.join(outDirectory, addFilenamePrefix( "small_", os.path.basename(filename)))
    im.save(newFilename, "JPEG")
    return newFilename

# Wrapper.
def delaunayFromPoints(points):
    start = time.perf_counter()
    triangles = delaunay(points)
    print("Delaunay-Triangulierung: %.2fs" % (time.perf_counter()-start))
    return triangles

# Wrapper.
def voronoiFromTriangles(triangles):
    start = time.perf_counter()
    polygons = createVoronoiFromDelaunay(triangles)
    print("Voronoi-Polygonalisierung: %.2fs" % (time.perf_counter()-start))
    return polygons
#增加的V-mask的代码------------------------------------------------------------------------------------------------------------
def point_in_polygon(x, y, polygon):
    """
    射线法判断点是否在多边形内
    polygon是[(x1,y1),(x2,y2),...]的顶点序列
    """
    num = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(num+1):
        p2x, p2y = polygon[i % num]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y)*(p2x - p1x)/(p2y - p1y)+p1x
                    else:
                        xinters = p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def brightenImage(im, factor):
    # 简单提高亮度的函数，factor>1越亮
    # 可以使用PIL的ImageEnhance模块，这里是示意
    # from PIL import ImageEnhance
    # enhancer = ImageEnhance.Brightness(im)
    # im_bright = enhancer.enhance(factor)
    # return im_bright
    
    # 如果不想使用ImageEnhance，可手动调整像素亮度：
    arr = np.array(im, dtype=np.float32)
    arr = arr * factor
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def polygon_gravity_center(pol):
    # 使用多边形重心公式（多边形为非自交）
    # 如果是简单多边形，可以用顶点平均值近似
    # 这里使用公式求质心
    # 参考：https://en.wikipedia.org/wiki/Centroid#Of_a_polygon
    area = 0.0
    cx = 0.0
    cy = 0.0
    n = len(pol)
    for i in range(n):
        x1, y1 = pol[i]
        x2, y2 = pol[(i+1) % n]
        cross = x1 * y2 - x2 * y1
        area += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    area = area / 2.0
    if area == 0:
        # 如果面积为0(几乎不可能，但防止除零错误)，用平均值代替
        xs = [p[0] for p in pol]
        ys = [p[1] for p in pol]
        return (sum(xs)/len(xs), sum(ys)/len(ys))
    cx = cx / (6.0 * area)
    cy = cy / (6.0 * area)
    return (cx, cy)

def getPolygonColorCount(pol, im):
    min_x = int(min(p[0] for p in pol))
    max_x = int(max(p[0] for p in pol))
    min_y = int(min(p[1] for p in pol))
    max_y = int(max(p[1] for p in pol))

    width, height = im.size
    black_count = 0
    white_count = 0

    for yy in range(min_y, max_y+1):
        if yy < 0 or yy >= height:
            continue
        for xx in range(min_x, max_x+1):
            if xx < 0 or xx >= width:
                continue
            if point_in_polygon(xx, yy, pol):
                pixel = im.getpixel((xx, yy))
                if isinstance(pixel, tuple):
                    val = pixel[0]
                else:
                    val = pixel
                if val < 128:
                    black_count += 1
                else:
                    white_count += 1
    return black_count, white_count
    # min_x = int(min(p[0] for p in pol))
    # max_x = int(max(p[0] for p in pol))
    # min_y = int(min(p[1] for p in pol))
    # max_y = int(max(p[1] for p in pol))

    # width, height = im.size
    # black_count = 0
    # white_count = 0

    # for yy in range(min_y, max_y+1):
    #     if yy < 0 or yy >= height:
    #         continue
    #     for xx in range(min_x, max_x+1):
    #         if xx < 0 or xx >= width:
    #             continue
    #         if point_in_polygon(xx, yy, pol):
    #             pixel = im.getpixel((xx, yy))
    #             if isinstance(pixel, tuple):
    #                 val = pixel[0]
    #             else:
    #                 val = pixel
    #             if val < 128:
    #                 black_count += 1
    #             else:
    #                 white_count += 1
    # return black_count, white_count

def fillPolygonWithVmask(pol, im, gx, mask_type):
    # mask_type: 0->V-mask0, 1->V-mask1
    # V-mask0: 左侧=1(白),右侧=0(黑)
    # V-mask1: 左侧=0(黑),右侧=1(白)
    (width, height) = im.size

    min_x = int(min(p[0] for p in pol))
    max_x = int(max(p[0] for p in pol))
    min_y = int(min(p[1] for p in pol))
    max_y = int(max(p[1] for p in pol))

    for yy in range(min_y, max_y+1):
        if yy < 0 or yy >= height:
            continue
        for xx in range(min_x, max_x+1):
            if xx < 0 or xx >= width:
                continue
            if point_in_polygon(xx, yy, pol):
                px = xx
                py = yy
                if px < gx:
                    # 重心线左侧
                    if mask_type == 0:
                        # V-mask0左侧=白
                        color = (255,255,255)
                    else:
                        # V-mask1左侧=黑
                        color = (0,0,0)
                else:
                    # 重心线右侧
                    if mask_type == 0:
                        # V-mask0右侧=黑
                        color = (0,0,0)
                    else:
                        # V-mask1右侧=白
                        color = (255,255,255)
                im.putpixel((px, py), color)
    

def drawVMasksImagesFromVoronoi(polygons, origIm, filename_prefix):
    # 本函数在已经有了Voronoi的polygons后调用
    start = time.perf_counter()
    (sizeX, sizeY) = origIm.size
    #  # 首先将origIm也放大到与Voronoi相同大小，以便坐标对应---------------------------
    # origImScaled = origIm.resize((sizeX*multiplier, sizeY*multiplier), Image.BICUBIC)
    #  # 创建与Voronoi同尺寸的Key Images
    # im_key1 = Image.new('RGB', (sizeX*multiplier, sizeY*multiplier), color=(255,255,255))
    # im_key2 = Image.new('RGB', (sizeX*multiplier, sizeY*multiplier), color=(255,255,255))

    # 创建两张与原图相同大小的图像（Key Image 1和Key Image 2）
    im_key1 = Image.new('RGB', (sizeX, sizeY), color=(255,255,255))
    im_key2 = Image.new('RGB', (sizeX, sizeY), color=(255,255,255))
    scaled_polygons = []
    for pol in polygons:
        scaled_pol = [(x*multiplier, y*multiplier) for (x,y) in pol]
        scaled_polygons.append(scaled_pol)
    # 遍历每个Voronoi区域并确定其mask
    for pol in polygons:
        if len(pol) < 3:
            continue
        # 这里改了origIm
        black_count, white_count = getPolygonColorCount(pol, origIm)
        if (black_count + white_count) == 0:
            # 默认白区
            s_r = 1
        else:
            if black_count > white_count:
                s_r = 0  # 黑
            else:
                s_r = 1  # 白

        gx, gy = polygon_gravity_center(pol)

        # 随机决定Key Image 1的mask类型
        key1_mask_type = 0 if random.random() < 0.5 else 1
        # 根据s_r决定Key Image 2的mask类型
        if s_r == 1:
            key2_mask_type = 1 - key1_mask_type
        else:
            key2_mask_type = key1_mask_type

        # 填充V-mask
        fillPolygonWithVmask(pol, im_key1, gx, key1_mask_type)
        fillPolygonWithVmask(pol, im_key2, gx, key2_mask_type)

    # 可按需要调亮图像（非必须）
    im_key1 = brightenImage(im_key1, 3.0)
    im_key2 = brightenImage(im_key2, 3.0)

    ImageFile.MAXBLOCK = im_key1.size[0] * im_key1.size[1]
    im_key1.save(filename_prefix + "_key1.jpg", "JPEG", quality=100, optimize=True, progressive=True)
    im_key2.save(filename_prefix + "_key2.jpg", "JPEG", quality=100, optimize=True, progressive=True)

    print("生成V-masks Key Images: %.2fs" % (time.perf_counter()-start))
    #这个是把两个生成的v-mask的图拼接起来
def overlay_vmask_images(key1_path, key2_path, output_path):
   
        im_key1 = Image.open(key1_path).convert("RGB")
        im_key2 = Image.open(key2_path).convert("RGB")

    # 确保两张图大小一致
        if im_key1.size != im_key2.size:
           raise ValueError("Key images must have the same dimensions.")

        width, height = im_key1.size
        im_result = Image.new('RGB', (width, height), color=(255,255,255))

        for y in range(height):
         for x in range(width):
            pixel1 = im_key1.getpixel((x,y))
            pixel2 = im_key2.getpixel((x,y))

            # 检查两张图在该像素的颜色
            # 假设黑色为(0,0,0)或接近全黑，白色为(255,255,255)
            # 如果任一图此像素为黑，则叠加后为黑，否则为白
            if pixel1[0] < 128 or pixel2[0] < 128:
                # 黑色
                im_result.putpixel((x,y), (0,0,0))
            else:
                # 白色
                im_result.putpixel((x,y), (255,255,255))

    # 保存叠加后的结果图像
        im_result.save(output_path, "JPEG", quality=100, optimize=True, progressive=True)
#这计算之间的差异
def calculate_difference_ratio(img1_path, img2_path):
    im1 = Image.open(img1_path).convert("RGB")
    im2 = Image.open(img2_path).convert("RGB")

    if im1.size != im2.size:
        raise ValueError("Images must have the same dimensions.")

    width, height = im1.size
    total_pixels = width * height
    diff_pixels = 0

    for y in range(height):
        for x in range(width):
            p1 = im1.getpixel((x, y))
            p2 = im2.getpixel((x, y))
            # 简单判断法：像素不完全相同就计为差异
            if p1 != p2:
                diff_pixels += 1

    return diff_pixels / float(total_pixels)
# ----------------- 以下为ε-采样与α-形状优化函数 -----------------
def epsilon_sampling(im, epsilon):
    width, height = im.size
    step = int(round(epsilon))
    if step < 1:
        step = 1
    points = []
    # 简单均匀采样
    for y in range(0, height, step):
        for x in range(0, width, step):
            points.append((x, y))
    return points

def point_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def construct_alpha_shape(points, alpha):
    # 简化版: 基于Delaunay，用边长<alpha的边作为α-形状边
    # triangles = delaunay(points)
    # edges = set()
    # for tri in triangles:
    #     (i1, i2, i3) = tri
    #     p1 = points[i1]
    #     p2 = points[i2]
    #     p3 = points[i3]

    #     if point_distance(p1, p2) < alpha:
    #         edges.add(tuple(sorted([i1, i2])))
    #     if point_distance(p2, p3) < alpha:
    #         edges.add(tuple(sorted([i2, i3])))
    #     if point_distance(p1, p3) < alpha:
    #         edges.add(tuple(sorted([i1, i3])))

    # # 转换为点坐标的边
    # alpha_edges = []
    # for e in edges:
    #     (i1, i2) = e
    #     alpha_edges.append((points[i1], points[i2]))
    # return alpha_edges
    raw_triangles = delaunay(points)
    point_index_map = {p: i for i, p in enumerate(points)}

    valid_triangles = []
    for tri in raw_triangles:
        coords = [p for p in tri if p is not None]
        if len(coords) == 3:
            try:
                i1, i2, i3 = (point_index_map[c] for c in coords)
                valid_triangles.append((i1, i2, i3))
            except KeyError:
                pass

    # 后面使用valid_triangles代替raw_triangles
    edges = set()
    for (i1,i2,i3) in valid_triangles:
        p1 = points[i1]
        p2 = points[i2]
        p3 = points[i3]
        if point_distance(p1,p2)<alpha:
            edges.add(tuple(sorted([i1,i2])))
        if point_distance(p2,p3)<alpha:
            edges.add(tuple(sorted([i2,i3])))
        if point_distance(p1,p3)<alpha:
            edges.add(tuple(sorted([i1,i3])))

    alpha_edges = [(points[e[0]], points[e[1]]) for e in edges]
    return alpha_edges

def is_shape_good_enough(alpha_edges, im):
    # 简单判定：如果边数量大于10就算够好（示例）
    return len(alpha_edges) > 10

def adjust_points_based_on_alpha_shape(alpha_edges, points):
    # 简单返回原点集（示例）
    return points

def alpha_shape_optimization(points, alpha_start, im, max_iter=5):
    current_alpha = alpha_start
    current_points = points[:]

    for i in range(max_iter):
        alpha_edges = construct_alpha_shape(current_points, current_alpha)
        if is_shape_good_enough(alpha_edges, im):
            # 尝试增大alpha简化
            new_alpha = current_alpha * 1.5
            new_edges = construct_alpha_shape(current_points, new_alpha)
            if is_shape_good_enough(new_edges, im):
                current_alpha = new_alpha
            else:
                # 增大后不好，回退
                break
        else:
            # 不够好，减少alpha增加细节
            new_alpha = current_alpha / 2.0
            if new_alpha < 1e-6:
                # 太小，尝试调整点集
                current_points = adjust_points_based_on_alpha_shape(alpha_edges, current_points)
                # 再试一次
                new_edges = construct_alpha_shape(current_points, new_alpha)
                if not is_shape_good_enough(new_edges, im):
                    break
                else:
                    current_alpha = new_alpha
            else:
                new_edges = construct_alpha_shape(current_points, new_alpha)
                if is_shape_good_enough(new_edges, im):
                    current_alpha = new_alpha
                else:
                    current_points = adjust_points_based_on_alpha_shape(alpha_edges, current_points)
                    current_alpha = new_alpha
    return current_points
#添加的代码---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Values
    parser.add_argument('-o', '--output', dest='output_filename', help='The filename to write the image to. Supported filetypes are BMP, TGA, PNG, and JPEG')
    parser.add_argument('-i', '--image-file', dest='input_filename', help='An image file to use when calculating triangle colors. Image dimensions will override dimensions set by -x and -y.')
    parser.add_argument('-f', '--factor', dest='factor',type=float, help='Factor definition. Determines the number of generated points (recommended value = 2.1 --> ~3000 points)')
    parser.add_argument('-r', '--random', dest='create_random', default=False, help='If enabled, set the points randomly.')
    parser.add_argument('-t', '--triangle', dest='create_triangle', default=True, help='If enabled, compute the triangle based in the spatial distribution of the image.')
    parser.add_argument('-v', '--voronoi', dest='create_voronoi', default=False, help='If enabled, compute the voronoi based in the spatial distribution of the image..')
    # 新增参数
    parser.add_argument('-hold','--threshold', dest='threshold', type=float, default=0.5, help='Allowed difference threshold (e.g. 0.5 means no more than 50% pixel difference).')
    parser.add_argument('-max','--max-iter', dest='max_iter', type=int, default=10, help='Max iteration if difference ratio is too large.')
    # 这两行代码来定义epsilon和alpha参数
    parser.add_argument('-e','--epsilon', type=float, default=5.0, help='Epsilon for epsilon-sampling')
    parser.add_argument('-a','--alpha', type=float, default=10.0, help='Initial alpha for alpha-shape')

    options = parser.parse_args()

    if(not os.path.isfile(options.input_filename)):
        print("There was an error in the path of the indicated file. Please check and try again!")
    else:
        (colorIm, blackIm) = loadAndFilterImage(options.input_filename)
        (width, height) = colorIm.size
        multiplier = 10

        if options.create_random:
            points = generateRandomPoints(15000, width, height)
            triangles = delaunayFromPoints(points)
            drawTriangulation(triangles, addFilenamePrefix( "random_", options.output_filename ), width, height, multiplier)

        if options.create_triangle:
            points = findPointsFromImage(blackIm, options.factor)
            triangles = delaunayFromPoints(points)
            drawImageColoredTriangles(triangles, addFilenamePrefix( "delaunay_", options.output_filename ), colorIm, multiplier)

        if options.create_voronoi:
        #     points = findPointsFromImage(blackIm, options.factor)
        #     triangles = delaunayFromPoints(points)
        #     polygons = voronoiFromTriangles(triangles)
        #     drawImageColoredVoronoi(polygons, addFilenamePrefix( "voronoi_", options.output_filename ), colorIm, multiplier)
        #     vmasks_filename_prefix = addFilenamePrefix("vmasks_", options.output_filename)
        #     drawVMasksImagesFromVoronoi(polygons, colorIm, vmasks_filename_prefix)
        #     key1_path = vmasks_filename_prefix + "_key1.jpg"
        #     key2_path = vmasks_filename_prefix + "_key2.jpg"
        #     overlay_output_path = vmasks_filename_prefix + "_overlay_result.jpg"
        #     overlay_vmask_images(key1_path, key2_path, overlay_output_path)

        # 我们在这里增加一个循环来尝试多次生成，直到满足threshold条件或超过max_iter
        # 按照正常尝试（增加factor） -> 不满足要求 -> 用ε-采样保证覆盖 -> 用α-形状迭代优化 -> 再比较这样的逻辑。
            current_factor = options.factor
            iteration = 0
            #第三次加了这里：
            satisfied = False

            while iteration < options.max_iter and not satisfied:
                iteration += 1
                print(f"Iteration {iteration}, using factor={current_factor}")

                # 重新用当前factor生成Voronoi和叠加图像
                points = findPointsFromImage(blackIm, current_factor)
                triangles = delaunayFromPoints(points)
                polygons = voronoiFromTriangles(triangles)

                voronoi_filename = addFilenamePrefix("voronoi_", options.output_filename)+ ".jpg"
                drawImageColoredVoronoi(polygons, voronoi_filename, colorIm, multiplier=1)

                vmasks_filename_prefix = addFilenamePrefix("vmasks_", options.output_filename)+ ".jpg"
                drawVMasksImagesFromVoronoi(polygons, colorIm, vmasks_filename_prefix)
                key1_path = vmasks_filename_prefix + "_key1.jpg"
                key2_path = vmasks_filename_prefix + "_key2.jpg"
                overlay_output_path = vmasks_filename_prefix + "_overlay_result.jpg"
                overlay_vmask_images(key1_path, key2_path, overlay_output_path)

                # 计算差异 (与Voronoi图比较，也可与原图比较，视需求而定)
                # 如果你想与原始输入图比较，则使用options.input_filename代替voronoi_filename即可
                diff_ratio = calculate_difference_ratio(options.input_filename, voronoi_filename)
                print(f"Difference ratio: {diff_ratio:.4f}")

                # 判断是否满足threshold
                if diff_ratio <= options.threshold:
                    # print("Difference ratio is acceptable. Stopping iteration.")
                    # break
                    # 第三次加的
                    print("Difference ratio is acceptable.")
                    satisfied = True
                else:
                    #  第三次加的：factor加大后仍不满足要求，尝试ε-采样+α-形状优化
                    print("Not satisfied. Trying epsilon_sampling + alpha_shape_optimization.")
                    eps_points = epsilon_sampling(blackIm, options.epsilon)
                    optimized_points = alpha_shape_optimization(eps_points, options.alpha, blackIm)
                    # 用优化后的点再次生成
                    triangles = delaunayFromPoints(optimized_points)
                    polygons = voronoiFromTriangles(triangles)
                    voronoi_filename = addFilenamePrefix("voronoi_optimized_", options.output_filename)+ ".jpg"
                    drawImageColoredVoronoi(polygons, voronoi_filename, colorIm, multiplier=1)
                    vmasks_filename_prefix = addFilenamePrefix("vmasks_optimized_", options.output_filename)+ ".jpg"
                    drawVMasksImagesFromVoronoi(polygons, colorIm, vmasks_filename_prefix)
                    key1_path = vmasks_filename_prefix + "_key1.jpg"
                    key2_path = vmasks_filename_prefix + "_key2.jpg"
                    overlay_output_path = vmasks_filename_prefix + "_overlay_result.jpg"
                    overlay_vmask_images(key1_path, key2_path, overlay_output_path)

                    diff_ratio = calculate_difference_ratio(options.input_filename, voronoi_filename)
                    print(f"Difference ratio after optimization: {diff_ratio:.4f}")

                    if diff_ratio <= options.threshold:
                        print("Difference ratio acceptable after optimization.")
                        satisfied = True
                    else:
                        print("Still not satisfied after optimization. Increasing factor and retrying.")
                        current_factor += 2.0

            if not satisfied:
                print("Reached maximum iterations without meeting threshold.")
                    # print("Difference ratio too large, increasing factor and retrying.")
                    # current_factor += 2.0
                    # if iteration >= options.max_iter:
                    #     print("Reached maximum iterations without meeting threshold. Stopping.")
                    #     break
        #autocontrastImage(addFilenamePrefix( "voronoi_", options.output_filename))
        #autocontrastImage(addFilenamePrefix("delaunay_", options.output_filename))
        #equalizeImage(addFilenamePrefix("voronoi_", filename))




















