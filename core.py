from pandas import read_csv, np
from matplotlib import pyplot as plt

DATA_SET_COLS = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y']


def extract_images(image):
    return image.apply(lambda image: np.fromstring(image, sep=' '))


def get_image_values(image):
    return np.vstack(image.values).astype(np.float32).reshape(-1, 96, 96, 1)


def load_training_dataframe(location: str = "datasets/training.csv"):
    """
    :param location:
    :return: tuple (x.y) where x is image array and y is the predictions
    """
    df = read_csv(location)
    df['Image'] = extract_images(df['Image'])
    df = df[DATA_SET_COLS + ['Image']]
    df = df.dropna()
    x_train = get_image_values(df['Image'])
    y_train = df[DATA_SET_COLS]
    y_train = y_train.values.astype(np.float32)
    return x_train, y_train


def load_testing_dataframe(location: str = "datasets/test.csv"):
    """
    :param location:
    :return:  x where x is image array
    """
    df = read_csv(location)
    df['Image'] = extract_images(df['Image'])
    df = df.dropna()
    x_train = get_image_values(df['Image'])
    return x_train


def view_image(image_arr, xs=None):
    arr = np.array(image_arr, dtype=np.uint8)
    arr.resize((96, 96))
    plt.imshow(arr, cmap='gray')
    for i in range(0, len(xs) - 1, 2):
        plt.scatter(xs[i], xs[i + 1], s=200, facecolors='none', edgecolors='r')
    plt.show()
