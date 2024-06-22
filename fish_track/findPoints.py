import cv2
import numpy as np
global xy, rmouse

# 可用滑鼠在影像中按左鍵讀取座標，並傳回最後的maxp個(預設4個)點。
# 按滑鼠右鍵或ESC鍵結束選取點
#--------------------------------------------
def onMouse(event,x,y,flags,param): 
    global xy, rmouse
    if event == cv2.EVENT_LBUTTONDOWN:    # 按滑鼠左鍵  
        xy = [x,y]
    elif event == cv2.EVENT_RBUTTONDOWN : # 按滑鼠右鍵
        rmouse = 1  
#--------------------------------------------
def getPoints(img, maxp=4):
    '''
    - Use mouse left click to select multipoint coordinates, 
    right click or ESC to stop selection. 
    - if too many points are selected, only the last maxp-points
    are choosed
    
    - return : (x,y)coordinate of mouse selecting points
    - parameters:
        img: source image
        maxp: maximum number of selecting points
    '''
    global xy, rmouse
    cv2.namedWindow('to select points')
    cv2.setMouseCallback('to select points', onMouse)

    xy = [] # 滑鼠的x,y座標 
    pts = [] # 存放所有滑鼠右鍵按下時的x,y座標
    rmouse = 0 # 滑鼠右鍵按下的flag
    maxpt = maxp # 最多傳回的座標點數
    imgcopy = img.copy()

    while True:
        cv2.imshow('to select points',imgcopy)
        key = cv2.waitKey(1)
        if key == 27 or rmouse == 1: # Esc鍵或滑鼠右鍵按下，迴路跳出
            if len(pts)> maxpt : # 若按下超過maxpt點座標
                pts = pts[-maxpt:] # 只保留最後的maxpt點座標
            break
        if xy : # 滑鼠左鍵被按下
            cv2.circle(imgcopy, tuple(xy), 6, [0,0,255], 1) # 畫大圓點
            cv2.circle(imgcopy, tuple(xy), 3, [255,0,0], 1) # 畫小圓點
            pts.append(xy) #左鍵x,y座標加入
            xy =[] #清空xy
    cv2.destroyWindow('to select points')
    return pts

#--------------------------------------------
def draw(winTitle, img, x=30, y=30 ):
    '''
    display single image on window
    param:
        winTitle: window title
        img: image to be display
        x: window position x-coordinate(=30)
        y: window position y-coordinate(=30)    
    '''
    # cv2.namedWindow(winTitle, cv2.WINDOW_NORMAL)
    cv2.namedWindow(winTitle)
    cv2.moveWindow(winTitle, x, y)
    cv2.imshow(winTitle, img)
    cv2.waitKey(-1)
    cv2.destroyWindow(winTitle)
#--------------------------------------------
def drawH(winTitle, img, x=30, y=30 ):
    '''
    display multiple images horizontally on window
    param:
        winTitle: window title
        img: (image1, image2, ...) images tuple
        x: window position x-coordinate(=30)
        y: window position y-coordinate(=30)    
    '''
    # cv2.namedWindow(winTitle, cv2.WINDOW_NORMAL)
    cv2.namedWindow(winTitle)
    cv2.moveWindow(winTitle, x, y)
    if isinstance(img, tuple):    
        imgH = img[0]
        for i in range(len(img)-1):
            imgH = np.hstack((imgH, img[i+1]))
    else:
        imgH = img
        
    cv2.imshow(winTitle, imgH)    
    cv2.waitKey(-1)
    cv2.destroyWindow(winTitle)

#--------------------------------------------
def main():
    img = cv2.imread('money.jpg') # 讀入影像
    print(getPoints(img,4))  # 滑鼠取影像上最後四點的座標
    drawH('3 images', (img, img, img)) # 水平顯示三張影像
#--------------------------------------------    
if __name__ =='__main__' :
    main()   
   
