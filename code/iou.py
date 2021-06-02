## Rotated IOU

import cv2

def iou_rotate(box_1, box_2):

    area1 = box_1[2] * box_1[3] ## area of the box_1
    area2 = box_2[2] * box_2[3]
    rect1 = ((box_1[0], box_1[1]), (box_1[2], box_1[3]), box_1[4]) ## centerx, centery, w, h, theta
    rect2 = ((box_2[0], box_2[1]), (box_2[2], box_2[3]), box_2[4])
    int_pts = cv2.rotatedRectangleIntersection(rect1, rect2)[1]
    
    if int_pts is not None:
        order_pts = cv2.convexHull(int_pts, returnPoints=True)
        int_area = cv2.contourArea(order_pts)
        
        iou = int_area * 1.0 / (area1 + area2 - int_area)
    else:
        iou = 0
    return iou