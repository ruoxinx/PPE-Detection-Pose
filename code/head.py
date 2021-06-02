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

def loc_head(img, temp):
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

        x_nose = keypoints['pose_keypoints_2d'][0]
        y_nose = keypoints['pose_keypoints_2d'][1]
        c_nose = keypoints['pose_keypoints_2d'][2]

        x_LEar = keypoints['pose_keypoints_2d'][3 * 17]
        y_LEar = keypoints['pose_keypoints_2d'][3 * 17 + 1]
        c_LEar = keypoints['pose_keypoints_2d'][3 * 17 + 2]

        # print(x_LEar, y_LEar, c_LEar)

        x_REar = keypoints['pose_keypoints_2d'][3 * 16]
        y_REar = keypoints['pose_keypoints_2d'][3 * 16 + 1]
        c_REar = keypoints['pose_keypoints_2d'][3 * 16 + 2]

        x_neck = keypoints['pose_keypoints_2d'][3 * 1]
        y_neck = keypoints['pose_keypoints_2d'][3 * 1 + 1]
        c_neck = keypoints['pose_keypoints_2d'][3 * 1 + 2]

        x_REye = keypoints['pose_keypoints_2d'][3 * 14]
        y_REye = keypoints['pose_keypoints_2d'][3 * 14 + 1]
        c_REye = keypoints['pose_keypoints_2d'][3 * 14 + 2]

        x_LEye = keypoints['pose_keypoints_2d'][3 * 15]
        y_LEye = keypoints['pose_keypoints_2d'][3 * 15 + 1]
        c_LEye = keypoints['pose_keypoints_2d'][3 * 15 + 2]

        num_key = np.nonzero(cordi_key)[0]
        ##print(len(num_key)/3)

        if len(num_key) / 3 <= 4:
            n = n - 1
            warning_txt = str(i) + '_' + 'partial visible'
            print(warning_txt)
            continue

        if c_LEar != 0 and c_REar != 0 and c_nose!=0:
            ang_1 = (y_LEar - y_REar) / (x_LEar - x_REar)  ## LEar and REar
            angle1 = -np.arctan(ang_1)
            angle1 = np.degrees(angle1)
            # print(angle1)

            theta = angle1

            n = 2
            x_mean = (x_REar + x_LEar) / n
            y_mean = (y_REar + y_LEar) / n

            dist = distance.euclidean([x_REar, y_REar], [x_LEar, y_LEar])

            L = 1.2 * dist

            x0 = int(x_mean - L / 2 * np.cos(theta * np.pi / 180))
            y0 = int(y_mean + L / 2 * np.sin(theta * np.pi / 180))

            x1 = int(x_mean + L / 2 * np.cos(theta * np.pi / 180))
            y1 = int(y_mean - L / 2 * np.sin(theta * np.pi / 180))

            x2 = int(x1 - L * np.sin(theta * np.pi / 180))
            y2 = int(y1 - L * np.cos(theta * np.pi / 180))

            x3 = int(x0 - L * np.sin(theta * np.pi / 180))
            y3 = int(y0 - L * np.cos(theta * np.pi / 180))

            # cv2.line(img, (x0, y0), (x1, y1), (0, 0, 255), 2)  # red
            # cv2.line(img, (x2, y2), (x3, y3), (0, 0, 255), 2)  # red
            #
            # cv2.line(img, (x0, y0), (x3, y3), (255, 0, 0), 2)  # blue
            # cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # blue

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

        elif c_LEar != 0 and c_REar != 0 and c_nose==0:
            ang_1 = (y_LEar - y_REar) / (x_LEar - x_REar)  ## LEar and REar
            angle1 = -np.arctan(ang_1)
            angle1 = np.degrees(angle1)
            # print(angle1)

            theta = angle1

            n = 2
            x_mean = (x_REar + x_LEar) / n
            y_mean = (y_REar + y_LEar) / n

            dist = distance.euclidean([x_REar, y_REar], [x_LEar, y_LEar])

            L = 1.2 * dist

            x0 = int(x_mean - L / 2 * np.cos(theta * np.pi / 180))
            y0 = int(y_mean + L / 2 * np.sin(theta * np.pi / 180))

            x1 = int(x_mean + L / 2 * np.cos(theta * np.pi / 180))
            y1 = int(y_mean - L / 2 * np.sin(theta * np.pi / 180))

            x2 = int(x1 - L * np.sin(theta * np.pi / 180))
            y2 = int(y1 - L * np.cos(theta * np.pi / 180))

            x3 = int(x0 - L * np.sin(theta * np.pi / 180))
            y3 = int(y0 - L * np.cos(theta * np.pi / 180))

            # cv2.line(img, (x0, y0), (x1, y1), (0, 0, 255), 2)  # red
            # cv2.line(img, (x2, y2), (x3, y3), (0, 0, 255), 2)  # red
            #
            # cv2.line(img, (x0, y0), (x3, y3), (255, 0, 0), 2)  # blue
            # cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # blue

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

        elif c_LEar != 0 and c_REar == 0 and c_nose != 0:
            ang_2 = (y_LEar - y_nose) / (x_LEar - x_nose)  ## LEar and REar
            angle2 = -np.arctan(ang_2)
            angle2 = np.degrees(angle2)

            theta = angle2

            if c_neck != 0:
                L = distance.euclidean([x_neck, y_neck], [x_nose, y_nose])
                L = max(L, 1.5 * distance.euclidean([x_LEar, y_LEar], [x_nose, y_nose]))

            else:
                L = 1.5 * distance.euclidean([x_LEar, y_LEar], [x_nose, y_nose])

            x0 = int(x_nose)
            y0 = int(y_nose)

            x1 = int(x_nose + L * np.cos(theta * np.pi / 180))
            y1 = int(y_nose - L * np.sin(theta * np.pi / 180))

            # print(np.cos(angle2*np.pi/180), np.sin(angle2*np.pi/180))

            x2 = int(x1 - L * np.sin(theta * np.pi / 180))
            y2 = int(y1 - L * np.cos(theta * np.pi / 180))

            x3 = int(x0 - L * np.sin(theta * np.pi / 180))
            y3 = int(y0 - L * np.cos(theta * np.pi / 180))

            # cv2.line(img, (x0, y0), (x1, y1), (0, 0, 255), 2)  # red
            # cv2.line(img, (x2, y2), (x3, y3), (0, 0, 255), 2)  # red
            #
            # cv2.line(img, (x0, y0), (x3, y3), (255, 0, 0), 2)  # blue
            # cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # blue

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


        elif c_REar != 0 and c_LEar == 0 and c_nose != 0:

            ang_3 = (y_REar - y_nose) / (x_REar - x_nose)  ## LEar and REar
            angle3 = np.arctan(ang_3)
            angle3 = - np.degrees(angle3)
            theta = angle3

            if c_neck != 0:
                L = distance.euclidean([x_neck, y_neck], [x_nose, y_nose])
                L = max(L, 1.5 * distance.euclidean([x_REar, y_REar], [x_nose, y_nose]))

            else:
                L = 1.5 * distance.euclidean([x_REar, y_REar], [x_nose, y_nose])

            x0 = int(x_nose)
            y0 = int(y_nose)

            x1 = int(x_nose - L * np.cos(theta * np.pi / 180))
            y1 = int(y_nose + L * np.sin(theta * np.pi / 180))

            # print(np.cos(angle2*np.pi/180), np.sin(angle2*np.pi/180))

            x2 = int(x1 - L * np.sin(theta * np.pi / 180))
            y2 = int(y1 - L * np.cos(theta * np.pi / 180))

            x3 = int(x0 - L * np.sin(theta * np.pi / 180))
            y3 = int(y0 - L * np.cos(theta * np.pi / 180))

            # cv2.line(img, (x0, y0), (x1, y1), (0, 0, 255), 2)  # red
            # cv2.line(img, (x2, y2), (x3, y3), (0, 0, 255), 2)  # red
            #
            # cv2.line(img, (x0, y0), (x3, y3), (255, 0, 0), 2)  # blue
            # cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # blue

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

        elif c_REar != 0 and c_LEar == 0 and c_nose == 0:

            L = distance.euclidean([x_REar, y_REar], [x_neck, y_neck])

            x0 = int(x_REar)
            y0 = int(y_REar)

            x1 = int(x0 - L)
            y1 = int(y0)

            # print(np.cos(angle2*np.pi/180), np.sin(angle2*np.pi/180))

            x2 = int(x1)
            y2 = int(y1 - L)

            x3 = int(x0)
            y3 = int(y0 - L)

            # cv2.line(img, (x0, y0), (x1, y1), (0, 0, 255), 2)  # red
            # cv2.line(img, (x2, y2), (x3, y3), (0, 0, 255), 2)  # red
            #
            # cv2.line(img, (x0, y0), (x3, y3), (255, 0, 0), 2)  # blue
            # cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # blue

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

        elif c_LEar != 0 and c_REar == 0 and c_nose == 0:

            L = distance.euclidean([x_LEar, y_LEar], [x_neck, y_neck])

            x0 = int(x_LEar)
            y0 = int(y_LEar)

            x1 = int(x0 + L)
            y1 = int(y0)

            # print(np.cos(angle2*np.pi/180), np.sin(angle2*np.pi/180))

            x2 = int(x1)
            y2 = int(y1 - L)

            x3 = int(x0)
            y3 = int(y0 - L)

            # cv2.line(img, (x0, y0), (x1, y1), (0, 0, 255), 2)  # red
            # cv2.line(img, (x2, y2), (x3, y3), (0, 0, 255), 2)  # red
            #
            # cv2.line(img, (x0, y0), (x3, y3), (255, 0, 0), 2)  # blue
            # cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # blue

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

        else:
            warning_txt = str(i) + '_'+'head invisible'
            print(warning_txt)

    cv2.imshow('img', img)
    cv2.waitKey(0)