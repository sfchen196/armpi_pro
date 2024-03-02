#!/usr/bin/python3
# coding=utf8
# Date:2022/05/30
import sys
import cv2
import math
import time
import rospy
import numpy as np
from threading import RLock, Timer, Thread

from std_srvs.srv import *
from std_msgs.msg import *
from sensor_msgs.msg import Image

from sensor.msg import Led
from visual_processing.msg import Result
from visual_processing.srv import SetParam
from hiwonder_servo_msgs.msg import MultiRawIdPosDur
from chassis_control.msg import SetTranslation, SetVelocity

from armpi_pro import PID
from armpi_pro import Misc
from armpi_pro import bus_servo_control
from kinematics import ik_transform

# 智能码垛

lock = RLock()
ik = ik_transform.ArmIK()

x_dis = 500
y_dis = 0.15
img_w = 640
img_h = 480
centreX = 320
centreY = 410
stack_en = False
steadier = False
__isRunning = False
object_id = 'None'
object_angle = 0
object_center_x = 0
object_center_y = 0
x_pid = PID.PID(P=0.06, I=0, D=0)    # pid初始化
y_pid = PID.PID(P=0.00003, I=0, D=0)

range_rgb = {
    'red':   (0, 0, 255),
    'blue':  (255, 0, 0),
    'green': (0, 255, 0),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
}

# 初始位置
def initMove(delay=True):
    with lock:
        target = ik.setPitchRanges((0, 0.15, 0.0), -180, -180, 0) # 逆运动学求解
        if target:
            servo_data = target[1]
            bus_servo_control.set_servos(joints_pub, 1800, ((1, 200), (2, 500), (3, servo_data['servo3']), (4, servo_data['servo4']),
                                                                                (5, servo_data['servo5']),(6, servo_data['servo6'])))
    if delay:
        rospy.sleep(2)

def turn_off_rgb():
    led = Led()
    led.index = 0
    led.rgb.r = 0
    led.rgb.g = 0
    led.rgb.b = 0
    rgb_pub.publish(led)
    led.index = 1
    rgb_pub.publish(led)

# 变量重置
def reset():
    global stack_en
    global steadier
    global object_id
    global x_dis, y_dis

    with lock:
        x_dis = 500
        y_dis = 0.15
        x_pid.clear()
        y_pid.clear()
        turn_off_rgb()
        stack_en = False
        steadier = False
        object_id = 'None'
        

# 初始化
def init():
    rospy.loginfo("intelligent palletizer Init")
    initMove()
    reset()


# 移动控制函数
def move():
    global stack_en
    global steadier
    global object_id
    global x_dis, y_dis
    
    count_ = 0
    offset_y = 0
    stack_num = 0
    start_en = True
    place_coord = {1:(0.18, 0.0, -0.09),
                   2:(0.18, 0.0, -0.05),
                   3:(0.18, 0.0, -0.02)}
    while __isRunning:
        if steadier and object_center_x > 0 and object_center_y > 0: 
            # 木块已经放稳，进行追踪夹取
            diff_x = abs(object_center_x - centreX)
            diff_y = abs(object_center_y - centreY)
            # X轴PID追踪
            if diff_x < 10:
                x_pid.SetPoint = object_center_x  # 设定
            else:
                x_pid.SetPoint = centreX
                
            x_pid.update(object_center_x) # 当前
            dx = x_pid.output             # 输出
            x_dis += int(dx)     
            x_dis = 200 if x_dis < 200 else x_dis
            x_dis = 800 if x_dis > 800 else x_dis
            # Y轴PID追踪
            if diff_y < 10:
                y_pid.SetPoint = object_center_y  # 设定
            else:
                y_pid.SetPoint = centreY
                
            y_pid.update(object_center_y) # 当前
            dy = y_pid.output             # 输出
            y_dis += dy  
            y_dis = 0.12 if y_dis < 0.12 else y_dis
            y_dis = 0.28 if y_dis > 0.28 else y_dis
            
            
            if start_en:
                start_en = False
                move_time = 500
            else:
                move_time = 20
            # 机械臂追踪移动到木块上方
            target = ik.setPitchRanges((0, round(y_dis, 4), 0.0), -180, -180, 0)
            if target:
                servo_data = target[1]
                bus_servo_control.set_servos(joints_pub, move_time,((3, servo_data['servo3']),(4, servo_data['servo4']),
                                                             (5, servo_data['servo5']), (6, x_dis)))
                rospy.sleep(move_time/1000)
                
            if abs(dx) < 3 and abs(dy) < 0.003 and not stack_en: # 等待机械臂稳定停在木块上方
                count_ += 1
                if count_ == 10:
                    count_ = 0
                    stack_en = True
                    angle = object_angle % 90
                    offset_y = Misc.map(target[2], -180, -150, -0.01, 0.02) # 设置位置补偿
            else:
                angle = 0
                count_ = 0
            
            if stack_en:
                angle_pul = Misc.map(angle, 0, 80, 500, 800)
                bus_servo_control.set_servos(joints_pub, 500, ((2, angle_pul),))
                rospy.sleep(0.5)
                bus_servo_control.set_servos(joints_pub, 500, ((1, 120),)) # 张开机械爪
                rospy.sleep(0.5)
                target = ik.setPitchRanges((0, round(y_dis + offset_y, 4), -0.08), -180, -180, 0) # 机械臂向下伸
                if target:
                    servo_data = target[1]
                    bus_servo_control.set_servos(joints_pub, 1000, ((3, servo_data['servo3']), (4, servo_data['servo4']),
                                                                    (5, servo_data['servo5']), (6, x_dis)))
                rospy.sleep(1.5)
                bus_servo_control.set_servos(joints_pub, 500, ((1, 450),)) # 闭合机械爪
                rospy.sleep(0.8)
                
                bus_servo_control.set_servos(joints_pub, 1500, ((1, 450), (2, 500), (3, 80), (4, 825), (5, 625), (6, 500))) # 机械臂抬起来
                rospy.sleep(1.5)
                
                stack_num += 1 # 码垛计量
                (x,y,z) = place_coord[stack_num]
                z += 0.03
                target = ik.setPitchRanges((x,y,z), -180, -180, 0) # 机械臂移动到色块放置位置上方
                if target:
                    servo_data = target[1]
                    bus_servo_control.set_servos(joints_pub, 1500, ((6, servo_data['servo6']),)) # 机械臂先转过去
                    rospy.sleep(1.2)
                    bus_servo_control.set_servos(joints_pub, 1600, ((3, servo_data['servo3']), (4, servo_data['servo4']), (5, servo_data['servo5']))) # 再放下了
                rospy.sleep(1.8)
                
                target = ik.setPitchRanges(place_coord[stack_num], -180, -180, 0) # 机械臂移动到色块放置位置
                if target:
                    servo_data = target[1]
                    bus_servo_control.set_servos(joints_pub, 1000, ((3, servo_data['servo3']), (4, servo_data['servo4']), (5, servo_data['servo5']))) # 再放下了
                rospy.sleep(1)

                bus_servo_control.set_servos(joints_pub, 500, ((1, 150),))  #张开机械爪
                rospy.sleep(0.8)
                
                if stack_num >= 3: # 码垛计量大于等于3，进行重置
                    stack_num = 0
                    
                #机械臂复位
                target = ik.setPitchRanges((0, 0.15, 0.0), -180, -180, 0)
                if target:
                    servo_data = target[1]
                    bus_servo_control.set_servos(joints_pub, 1000, ((1, 200), (2, 500), (3, servo_data['servo3']),
                                                                    (4, servo_data['servo4']), (5, servo_data['servo5'])))
                    rospy.sleep(1)
                    bus_servo_control.set_servos(joints_pub, 1500, ((6, servo_data['servo6']),))
                    rospy.sleep(1.5)
                
                start_en = True
                reset()  # 变量重置            
        else:
            rospy.sleep(0.01)

count = 0
last_x = 0
last_y = 0
# 图像处理结果回调函数
def run(msg):
    global count
    global steadier
    global object_id
    global object_angle
    global last_x, last_y
    global object_center_x
    global object_center_y   

    # 获得图像处理结果
    object_center_x = msg.center_x  # 目标中心X坐标
    object_center_y = msg.center_y  # 目标中心Y坐标
    object_angle = msg.angle        # 目标旋转角
    object_id = msg.data            # 标签id号
    
    if object_id != 0:
        if not steadier: # 判断木块是否放稳
            dx = abs(object_center_x - last_x)
            dy = abs(object_center_y - last_y)
            last_x = object_center_x
            last_y = object_center_y
            if dx < 3 and dy < 3:
                count += 1
                if count == 10:
                    count = 0
                    steadier = True # 放稳
            else:
                count = 0


result_sub = None
heartbeat_timer = None
# APP enter服务回调函数
def enter_func(msg):
    global lock
    global result_sub

    rospy.loginfo("enter intelligent palletizer")
    init()
    with lock:
        if result_sub is None:
            rospy.ServiceProxy('/visual_processing/enter', Trigger)()
            result_sub = rospy.Subscriber('/visual_processing/result', Result, run) # 订阅图像处理处理节点结果

    return [True, 'enter']

# APP exit服务回调函数
def exit_func(msg):
    global lock
    global result_sub
    global __isRunning
    global heartbeat_timer

    rospy.loginfo("exit intelligent palletizer")
    with lock:
        rospy.ServiceProxy('/visual_processing/exit', Trigger)()
        __isRunning = False
        reset()
        try:
            if result_sub is not None:
                result_sub.unregister() # 注销订阅图像处理结果
                result_sub = None
            if heartbeat_timer is not None:
                heartbeat_timer.cancel() # 关闭心跳包服务
                heartbeat_timer = None
        except BaseException as e:
            rospy.loginfo('%s', e)

    return [True, 'exit']

# 开始运行函数
def start_running():
    global lock
    global __isRunning

    rospy.loginfo("start running intelligent palletizer")
    with lock:
        __isRunning = True
        visual_running = rospy.ServiceProxy('/visual_processing/set_running', SetParam)
        visual_running('apriltag','')
        rospy.sleep(0.1)
        # 运行子线程
        th = Thread(target=move)
        th.setDaemon(True)
        th.start()

# 停止运行函数
def stop_running():
    global lock
    global __isRunning

    rospy.loginfo("stop running intelligent palletizer")
    with lock:
        reset()
        __isRunning = False
        initMove(delay=False)
        rospy.ServiceProxy('/visual_processing/set_running', SetParam)()

# APP running服务回调函数
def set_running(msg):
    if msg.data:
        start_running()
    else:
        stop_running()

    return [True, 'set_running']

# 设置扩展版RGB彩灯
def set_LED(color):
    global lock
    with lock:
        led = Led()
        led.index = 0
        led.rgb.r = range_rgb[color][2]
        led.rgb.g = range_rgb[color][1]
        led.rgb.b = range_rgb[color][0]
        rgb_pub.publish(led)
        rospy.sleep(0.05)
        led.index = 1
        rgb_pub.publish(led)
        rospy.sleep(0.05)

# APP heartbeat服务回调函数
def heartbeat_srv_cb(msg):
    global heartbeat_timer

    if isinstance(heartbeat_timer, Timer):
        heartbeat_timer.cancel()
    if msg.data:
        heartbeat_timer = Timer(5, rospy.ServiceProxy('/intelligent_palletizer/exit', Trigger))
        heartbeat_timer.start()
    rsp = SetBoolResponse()
    rsp.success = msg.data

    return rsp


if __name__ == '__main__':
    # 初始化节点
    rospy.init_node('intelligent_palletizer', log_level=rospy.DEBUG)
    # 舵机发布
    joints_pub = rospy.Publisher('/servo_controllers/port_id_1/multi_id_pos_dur', MultiRawIdPosDur, queue_size=1)
    # app通信服务
    enter_srv = rospy.Service('/intelligent_palletizer/enter', Trigger, enter_func)
    exit_srv = rospy.Service('/intelligent_palletizer/exit', Trigger, exit_func)
    running_srv = rospy.Service('/intelligent_palletizer/set_running', SetBool, set_running)
    heartbeat_srv = rospy.Service('/intelligent_palletizer/heartbeat', SetBool, heartbeat_srv_cb)
    # 麦轮底盘控制
    set_velocity = rospy.Publisher('/chassis_control/set_velocity', SetVelocity, queue_size=1)
    set_translation = rospy.Publisher('/chassis_control/set_translation', SetTranslation, queue_size=1)
    # 蜂鸣器
    buzzer_pub = rospy.Publisher('/sensor/buzzer', Float32, queue_size=1)
    # rgb 灯
    rgb_pub = rospy.Publisher('/sensor/rgb_led', Led, queue_size=1)
    rospy.sleep(0.5) # pub之后必须延时才能生效

    debug = False
    if debug:
        rospy.sleep(0.2)
        enter_func(1)
        start_running()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
