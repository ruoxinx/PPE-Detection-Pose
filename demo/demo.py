from PIL import Image
from yolo import YOLO
from frcnn import FRCNN
from ssd import SSD

if __name__ == "__main__":
    ## --------------------------------- #
    # Faster R-CNN model
    frcnn = FRCNN()
    img = Image.open('../examples/000122.jpg')
    frcnn_out = frcnn.detect_image(img)
    frcnn_out.show()

    ## --------------------------------- #
    # YOLO model
    yolo = YOLO()
    img = Image.open('../examples/000122.jpg')
    yolo_out = yolo.detect_image(img)
    yolo_out.show()

    ## --------------------------------- #
    # SSD model
    ssd = SSD()
    img = Image.open('../examples/000122.jpg')
    ssd_out = ssd.detect_image(img)
    ssd_out.show()