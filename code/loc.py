from upbody import loc_upbody
from head import loc_head

if __name__ == "__main__":
    ## load images and json files
    img = '../examples/000124.jpg'
    json = '../examples/000124_keypoints.json'
    loc_upbody(img, json)
    loc_head(img, json)