import cv2
import numpy as np
import mediapipe as mp
import math
import random

class SnakeGameClass:
    def __init__(self):
        self.points=[]             #all points of snake
        self.length=[]             #distance between each points
        self.currentLength=0       #total length of the snake
        self.allowedLength=150     #total allowed length
        self.previousHead=0,0      #previous head point
        
        
        
        self.foodPoint=0,0
        self.randomFoodLocation()
        self.score=0
        self.gameOver=False
        
        
    def randomFoodLocation(self):
        self.foodPoint=random.randint(100,500),random.randint(100,500)
        
    def update(self,imgMain,currentHead):
        if self.gameOver:
            cv2.putText(imgMain, "GAME OVER", (500,500 ),  cv2.FONT_HERSHEY_SIMPLEX, 1,(50, 50,50), 3)
            cv2.putText(imgMain, f'YOUR SCORE:{self.score}', (100,100 ),  cv2.FONT_HERSHEY_SIMPLEX, 1,(50, 50,50), 3)
            
            
        else:
            px,py=self.previousHead
            cx,cy=currentHead
        
            self.points.append([cx,cy])
            distance=math.hypot(cx-px,cy-py)
            self.length.append(distance)
        
            self.currentLength+=distance
            self.previousHead=cx,cy
        
        #length reduction
            if self.currentLength>self.allowedLength:
                for i,length in enumerate(self.length):
                    self.currentLength-=length
                    self.length.pop(i)
                    self.points.pop(i)
                
                    if self.currentLength<self.allowedLength:
                        break
            
        #check if snake eats the food
            rx,ry=self.foodPoint
            if rx-20<cx<rx+20 and ry-20<cy<ry+20:
                self.randomFoodLocation()
                self.allowedLength+=50
                self.score+=1
                print(self.score)
        
            
        #draw snake
            if self.points:
            
                for (i,points) in enumerate(self.points):
                    if i!=0:
                        cv2.line(imgMain,self.points[i-1],self.points[i],(0,0,255),20)
                    cv2.circle(frame,self.points[-1],20,(200,0,200),-1)
        #check for collision
            pts=np.array(self.points[:-2],np.int32)
            pts=pts.reshape((-1,1,2))
            cv2.polylines(imgMain,[pts],False,(0,200,0),3)
            minDist= cv2.pointPolygonTest(pts,(cx,cy),True)
   
            cv2.putText(imgMain, f'YOUR SCORE:{self.score}', (100,100 ),  cv2.FONT_HERSHEY_SIMPLEX, 1,(50, 50,50), 3)
            if -1<=minDist<=1:
                print("hit")
                self.gameOver=True
                self.points=[]             #all points of snake
                self.length=[]             #distance between each points
                self.currentLength=0       #total length of the snake
                self.allowedLength=150     #total allowed length
                self.previousHead=0,0      #previous head point
                self.randomFoodLocation()
        
        
        #draw food
        
            rx,ry=self.foodPoint
            imgMain=cv2.rectangle(imgMain,(rx-20,ry-20),(rx+20,ry+20),(255,0,0),-1)
        
        return imgMain
        
        
game=SnakeGameClass()  

cap=cv2.VideoCapture(0)

mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils
pointIndex=[]

while(True):
    
    ret,frame=cap.read()
    frame=cv2.resize(frame,(1080,720))
    frame=cv2.flip(frame,1)
    framergb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results=hands.process(framergb)
    if cv2.waitKey(1) & 0xFF==ord('w'):
        break
   
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList=[]                  
            for id,lm in enumerate(handLms.landmark):
                h,w,c=frame.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                lmList.append([id,cx,cy])
                    
            pointIndex=[lmList[8][1],lmList[8][2]]
            
            
            mpDraw.draw_landmarks(frame,handLms,mpHands.HAND_CONNECTIONS)
        cv2.circle(frame,(pointIndex[0],pointIndex[1]),5,(0,0,0),-1)
        frame=game.update(frame,pointIndex)  
            
    cv2.imshow('frame',frame)
cap.release()
cv2.destroyAllWindows()