from enum import Enum
import ast, sys

def get_label(label_name):
    ''' 
    取得標籤物件 
    '''
    label_file = 'labels/'
    if label_name.upper()=='COCO':
        label_file += 'coco80.txt'
    elif label_name.upper()=='WILL':
        label_file += 'will3.txt'
    elif label_name.upper()=='IMAGENET':
        label_file += 'imagenet1000.txt'
    elif label_name.upper()=='MAX':
        label_file += 'max.txt'
    elif label_name.upper()=='USB_INNODISK':
        label_file += 'usb_innodisk.txt'
    else:
        print('ERROR')
        sys.exit(1)

    with open(label_file, 'r') as f:
        cnt = f.read()
        label = ast.literal_eval(cnt)

    return label

class COCOLabels(Enum):
    PERSON = 0
    BICYCLE = 1
    CAR = 2
    MOTORBIKE = 3
    AEROPLANE = 4
    BUS = 5
    TRAIN = 6
    TRUCK = 7
    BOAT = 8
    TRAFFIC_LIGHT = 9
    FIRE_HYDRANT = 10
    STOP_SIGN = 11
    PARKING_METER = 12
    BENCH = 13
    BIRD = 14
    CAT = 15
    DOG = 16
    HORSE = 17
    SHEEP = 18
    COW = 19
    ELEPHANT = 20
    BEAR = 21
    ZEBRA = 22
    GIRAFFE = 23
    BACKPACK = 24
    UMBRELLA = 25
    HANDBAG = 26
    TIE = 27
    SUITCASE = 28
    FRISBEE = 29
    SKIS = 30
    SNOWBOARD = 31
    SPORTS_BALL = 32
    KITE = 33
    BASEBALL_BAT = 34
    BASEBALL_GLOVE = 35
    SKATEBOARD = 36
    SURFBOARD = 37
    TENNIS_RACKET = 38
    BOTTLE = 39
    WINE_GLASS = 40
    CUP = 41
    FORK = 42
    KNIFE = 43
    SPOON = 44
    BOWL = 45
    BANANA = 46
    APPLE = 47
    SANDWICH = 48
    ORANGE = 49
    BROCCOLI = 50
    CARROT = 51
    HOT_DOG = 52
    PIZZA = 53
    DONUT = 54
    CAKE = 55
    CHAIR = 56
    SOFA = 57
    POTTEDPLANT = 58
    BED = 59
    DININGTABLE = 60
    TOILET = 61
    TVMONITOR = 62
    LAPTOP = 63
    MOUSE = 64
    REMOTE = 65
    KEYBOARD = 66
    CELL_PHONE = 67
    MICROWAVE = 68
    OVEN = 69
    TOASTER = 70
    SINK = 71
    REFRIGERATOR = 72
    BOOK = 73
    CLOCK = 74
    VASE = 75
    SCISSORS = 76
    TEDDY_BEAR = 77
    HAIR_DRIER = 78
    TOOTHBRUSH = 79

class WillLabels(Enum):
    mask = 0
    bad = 1
    abnormal = 2

class MaxLabels(Enum):
    mask = 0
    non_mask = 1