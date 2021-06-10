from upbody import loc_upbody
from head import loc_head

def main(img, json):
    loc_upbody(img, json)
    loc_head(img, json)

if __name__ == "__main__":
    ## input images and json files
    img = '../examples/000124.jpg'
    temp = '../examples/000124_keypoints.json'
    main(img, temp)