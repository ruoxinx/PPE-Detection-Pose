import json
import cv2
import numpy as np
from scipy.spatial import distance

# //     {0,  "Nose"},
# //     {1,  "Neck"},
# //     {2,  "RShoulder"},
# //     {3,  "RElbow"},
# //     {4,  "RWrist"},
# //     {5,  "LShoulder"},
# //     {6,  "LElbow"},
# //     {7,  "LWrist"},
#
# //     {8,  "RHip"},
# //     {9, "RKnee"},
# //     {10, "RAnkle"},
# //     {11, "LHip"},
# //     {12, "LKnee"},
# //     {13, "LAnkle"},
# //     {14, "REye"},
# //     {15, "LEye"},
# //     {16, "REar"},
# //     {17, "LEar"},

def loc_upbody(img, temp):
    img = cv2.imread(img)
    with open(temp, 'r') as load_f:
        temp = json.load(load_f)
    num_workers = 0
    num_workers = len(temp['people'])
    n = num_workers
    color = (0, 255, 0)

    for i in range(num_workers):
        keypoints = temp['people'][i]

        cordi_key = keypoints['pose_keypoints_2d']

        x_neck = keypoints['pose_keypoints_2d'][3 * 1]
        y_neck = keypoints['pose_keypoints_2d'][3 * 1 + 1]
        c_neck = keypoints['pose_keypoints_2d'][3 * 1 + 2]

        x_RShoulder = keypoints['pose_keypoints_2d'][3 * 2]
        y_RShoulder = keypoints['pose_keypoints_2d'][3 * 2 + 1]
        c_RShoulder = keypoints['pose_keypoints_2d'][3 * 2 + 2]

        x_LShoulder = keypoints['pose_keypoints_2d'][3 * 5]
        y_LShoulder = keypoints['pose_keypoints_2d'][3 * 5 + 1]
        c_LShoulder = keypoints['pose_keypoints_2d'][3 * 5 + 2]

        x_LHip = keypoints['pose_keypoints_2d'][3 * 11]
        y_LHip = keypoints['pose_keypoints_2d'][3 * 11 + 1]
        c_LHip = keypoints['pose_keypoints_2d'][3 * 11 + 2]

        x_RHip = keypoints['pose_keypoints_2d'][3 * 8]
        y_RHip = keypoints['pose_keypoints_2d'][3 * 8 + 1]
        c_RHip = keypoints['pose_keypoints_2d'][3 * 8 + 2]

        num_key = np.nonzero(cordi_key)[0]


        if len(num_key) / 3 <= 4:
            n = n - 1
            warning_txt = str(i) + '_' + 'pose error'
            print(warning_txt)
            continue

        if c_LShoulder != 0 and c_RShoulder != 0 and c_LHip != 0 and c_RHip !=0:

            x_pelvis = (x_LHip + x_RHip) / 2
            y_pelvis = (y_LHip + y_RHip) / 2

            dist1 = distance.euclidean([x_LShoulder, y_LShoulder], [x_RShoulder, y_RShoulder])
            dist2 = distance.euclidean([x_neck, y_neck], [x_pelvis, y_pelvis])

            a = dist1 / dist2
            #print(a)

            L_trans = 0.6 * dist2
            L_x = max(dist1, L_trans)
            L_y = dist2

            ang_1 = (y_neck - y_pelvis) / (x_neck - x_pelvis)  ## LEar and REar
            angle1 = np.arctan(ang_1)
            angle1 = np.degrees(angle1)
            #print(angle1)
            if angle1 < 0:
                theta = angle1 + 180
            else:
                theta = angle1

            x0 = int(x_neck - L_x / 2 * np.sin(theta * np.pi / 180))
            y0 = int(y_neck + L_x / 2 * np.cos(theta * np.pi / 180))

            x1 = int(x_neck + L_x / 2 * np.sin(theta * np.pi / 180))
            y1 = int(y_neck - L_x / 2 * np.cos(theta * np.pi / 180))

            x2 = int(x_pelvis - L_x /2 * np.sin(theta * np.pi / 180))
            y2 = int(y_pelvis + L_x /2 * np.cos(theta * np.pi / 180))

            x3 = int(x_pelvis + L_x /2 * np.sin(theta * np.pi / 180))
            y3 = int(y_pelvis - L_x /2 * np.cos(theta * np.pi / 180))

            # cv2.line(img, (x0, y0), (x1, y1), (0, 0, 255), 20)  # red
            # cv2.line(img, (x2, y2), (x3, y3), (0, 0, 255), 20)  # red
            #
            # cv2.line(img, (x0, y0), (x2, y2), (255, 0, 0), 20)  # blue
            # cv2.line(img, (x1, y1), (x3, y3), (255, 0, 0), 20)  # blue

            cnt = np.array([
                [[x0, y0]],
                [[x1, y1]],
                [[x3, y3]],
                [[x2, y2]]
            ])

            rect = cv2.minAreaRect(cnt)

            # the order of the box points: bottom left, top left, top right,
            # bottom right
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            cv2.drawContours(img, [box], 0, (0, 255, 0), 2) ##RGB Green color

            # get width and height of the detected rectangle
            width = int(rect[1][0])
            height = int(rect[1][1])

            src_pts = box.astype("float32")
            # coordinate of the points in box points after the rectangle has been
            # straightened
            dst_pts = np.array([[0, height - 1],
                                [0, 0],
                                [width - 1, 0],
                                [width - 1, height - 1]], dtype="float32")

            # the perspective transformation matrix
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)

            # directly warp the rotated rectangle to get the straightened rectangle
            warped = cv2.warpPerspective(img, M, (width, height))

        elif c_LShoulder != 0 and c_RShoulder != 0 and (c_LHip == 0 or c_RHip ==0):
            # warning_txt = file_name[0:6] + '_' + str(i) + '_' + 'lower body invisible'
            # print(warning_txt)

            dist = distance.euclidean([x_LShoulder, y_LShoulder], [x_RShoulder, y_RShoulder])
            L_x = dist
            L_y = dist / 0.6

            ang_2 = (y_RShoulder - y_LShoulder) / (x_RShoulder - x_LShoulder)  ## LEar and REar
            angle2 = -np.arctan(ang_2)
            angle2 = np.degrees(angle2)

            theta = angle2

            x0 = int(x_LShoulder)
            y0 = int(y_LShoulder)

            x1 = int(x_RShoulder)
            y1 = int(y_RShoulder)

            x2 = int(x0 + L_y * np.sin(theta * np.pi / 180))
            y2 = int(y0 + L_y * np.cos(theta * np.pi / 180))

            x3 = int(x1 + L_y * np.sin(theta * np.pi / 180))
            y3 = int(y1 + L_y * np.cos(theta * np.pi / 180))

            # cv2.line(img, (x0, y0), (x1, y1), (0, 0, 255), 20)  # red  横线
            # cv2.line(img, (x2, y2), (x3, y3), (0, 0, 255), 20)  # red  横线
            #
            # cv2.line(img, (x0, y0), (x2, y2), (255, 0, 0), 20)  # blue    竖线
            # cv2.line(img, (x1, y1), (x3, y3), (255, 0, 0), 20)  # blue    竖线

            cnt = np.array([
                [[x0, y0]],
                [[x1, y1]],
                [[x3, y3]],
                [[x2, y2]]
            ])

            rect = cv2.minAreaRect(cnt)

            # the order of the box points: bottom left, top left, top right,
            # bottom right
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

            # get width and height of the detected rectangle
            width = int(rect[1][0])
            height = int(rect[1][1])

            src_pts = box.astype("float32")
            # coordinate of the points in box points after the rectangle has been
            # straightened
            dst_pts = np.array([[0, height - 1],
                                [0, 0],
                                [width - 1, 0],
                                [width - 1, height - 1]], dtype="float32")

            # the perspective transformation matrix
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)

            # directly warp the rotated rectangle to get the straightened rectangle
            warped = cv2.warpPerspective(img, M, (width, height))

        elif c_LShoulder != 0 and c_RShoulder == 0 and c_LHip != 0:

            dist = distance.euclidean([x_LShoulder, y_LShoulder], [x_LHip, y_LHip])
            L_y = dist
            L_x = dist * 0.6

            ang_2 = (y_LShoulder - y_LHip) / (x_LShoulder - x_LHip)  ## LEar and REar
            angle2 = -np.arctan(ang_2)
            angle2 = np.degrees(angle2)

            theta = angle2

            x0 = int(x_LShoulder)
            y0 = int(y_LShoulder)

            x1 = int(x_LHip)
            y1 = int(y_LHip)

            x2 = int(x0 - L_x * np.sin(theta * np.pi / 180))
            y2 = int(y0 - L_x * np.cos(theta * np.pi / 180))

            x3 = int(x1- L_x * np.sin(theta * np.pi / 180))
            y3 = int(y1 - L_x * np.cos(theta * np.pi / 180))

            # cv2.line(img, (x0, y0), (x2, y2), (0, 0, 255), 2)  # red  横线
            # cv2.line(img, (x1, y1), (x3, y3), (0, 0, 255), 2)  # red  横线
            #
            # cv2.line(img, (x0, y0), (x1, y1), (255, 0, 0), 2)  # blue    竖线
            # cv2.line(img, (x2, y2), (x3, y3), (255, 0, 0), 2)  # blue    竖线

            cnt = np.array([
                [[x0, y0]],
                [[x1, y1]],
                [[x3, y3]],
                [[x2, y2]]
            ])

            rect = cv2.minAreaRect(cnt)

            # the order of the box points: bottom left, top left, top right,
            # bottom right
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

            # get width and height of the detected rectangle
            width = int(rect[1][0])
            height = int(rect[1][1])

            src_pts = box.astype("float32")
            # coordinate of the points in box points after the rectangle has been
            # straightened
            dst_pts = np.array([[0, height - 1],
                                [0, 0],
                                [width - 1, 0],
                                [width - 1, height - 1]], dtype="float32")

            # the perspective transformation matrix
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)

            # directly warp the rotated rectangle to get the straightened rectangle
            warped = cv2.warpPerspective(img, M, (width, height))

        elif c_RShoulder != 0 and c_LShoulder == 0 and c_RHip != 0:

            dist = distance.euclidean([x_RShoulder, y_RShoulder], [x_RHip, y_RHip])
            L_y = dist
            L_x = dist * 0.6

            ang_2 = (y_RShoulder - y_RHip) / (x_RShoulder - x_RHip)  ## LEar and REar
            angle2 = -np.arctan(ang_2)
            angle2 = np.degrees(angle2)

            theta = angle2

            x0 = int(x_RShoulder)
            y0 = int(y_RShoulder)

            x1 = int(x_RHip)
            y1 = int(y_RHip)

            x2 = int(x0 - L_x * np.sin(theta * np.pi / 180))
            y2 = int(y0 - L_x * np.cos(theta * np.pi / 180))

            x3 = int(x1 - L_x * np.sin(theta * np.pi / 180))
            y3 = int(y1 - L_x * np.cos(theta * np.pi / 180))

            # cv2.line(img, (x0, y0), (x2, y2), (0, 0, 255), 20)  # red
            # cv2.line(img, (x1, y1), (x3, y3), (0, 0, 255), 20)  # red
            #
            # cv2.line(img, (x0, y0), (x1, y1), (255, 0, 0), 20)  # blue
            # cv2.line(img, (x2, y2), (x3, y3), (255, 0, 0), 20)  # blue

            cnt = np.array([
                [[x0, y0]],
                [[x1, y1]],
                [[x3, y3]],
                [[x2, y2]]
            ])

            rect = cv2.minAreaRect(cnt)

            # the order of the box points: bottom left, top left, top right,
            # bottom right
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

            # get width and height of the detected rectangle
            width = int(rect[1][0])
            height = int(rect[1][1])

            src_pts = box.astype("float32")
            # coordinate of the points in box points after the rectangle has been
            # straightened
            dst_pts = np.array([[0, height - 1],
                                [0, 0],
                                [width - 1, 0],
                                [width - 1, height - 1]], dtype="float32")

            # the perspective transformation matrix
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)

            # directly warp the rotated rectangle to get the straightened rectangle
            warped = cv2.warpPerspective(img, M, (width, height))

        else:
            warning_txt = str(i) + '_' + 'body invisible'
            print(warning_txt)

    cv2.imshow('img', img)
    cv2.waitKey(0)