from upbody import  loc_upbody
from head import  loc_head

def main():

    ## input images and json files
    img = '../examples/000124.jpg'
    temp = '../examples/000124_keypoints.json'

    loc_upbody(img, temp)
    loc_head(img, temp)

if __name__ == "__main__":
    main()
