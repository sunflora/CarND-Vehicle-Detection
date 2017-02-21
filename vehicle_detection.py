import cv2
import numpy as np
import glob
import time

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from scipy.ndimage.measurements import label

from detection_training import DetectionTraining
from training_parameters import TrainingParameters
from sliding_window_search import SlidingWindowSearch
from search_parameters import WindowSearchParameters

from moviepy.editor import VideoFileClip

WRITEUP = True
DEBUG = not True
cars = []
noncars = []
car_features = []
noncar_features = []
NONCARS_LABEL = 0
CARS_LABEL = 1
svc = None
X_scaler = None
heatmap_history = []
HEATMAP_HISTORY_SIZE = 15
HEATMAP_HISTORICAL_RATIO = 0.9

def visualize(fig, rows, cols, imgs, titles):
    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i+1)
        plt.title(i+1)
        img_dims = len(img.shape)
        if img_dims < 3:
            plt.imshow(img, cmap='hot')
            plt.title(titles[i])
        else:
            plt.imshow(img)
            plt.title(titles[i])
    filename = "output_images/test_images_and_heatmaps.png"
    plt.savefig(filename)
    plt.close()

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

def retrieve_labeled_data():
    global cars, noncars

    print("Begin retrieving images...")
    cars = glob.glob('labeled_data/vehicles/*/*.png')
    print("--- Number of car images: ", len(cars))
    print("--- Shape of car images:", mpimg.imread(cars[0]).shape)

    noncars = glob.glob('labeled_data/non-vehicles/*/*.png')
    print("--- Number of noncar images: ", len(noncars))
    print("--- Shape of noncar images:", mpimg.imread(noncars[0]).shape)


def plot_sample_images():
    # Just for fun choose random car / non-car indices and plot example images
    if DEBUG: print("len(cars): ", len(cars))
    car_ind = np.random.randint(0, len(cars))
    noncars_ind = np.random.randint(0, len(noncars))

    # Read in car / not-car images
    car_image = mpimg.imread(cars[car_ind])
    noncar_image = mpimg.imread(noncars[noncars_ind])

    # Plot the examples
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(car_image)
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(noncar_image)
    plt.title('Example Non-car Image')
    plt.savefig(filename="output_images/01-sample-images.png")

def plot_hog_features_of_sample_images():
    # Just for fun choose random car / non-car indices and plot example images
    if DEBUG: print("len(cars): ", len(cars))
    car_ind = np.random.randint(0, len(cars))
    noncars_ind = np.random.randint(0, len(noncars))

    # Read in car / not-car images
    car_image = mpimg.imread(cars[car_ind])
    noncar_image = mpimg.imread(noncars[noncars_ind])
    if DEBUG: print("max car_image: ", np.amax(car_image))

    car_sample = cv2.cvtColor(car_image, cv2.COLOR_RGB2YCrCb)
    noncar_sample = cv2.cvtColor(noncar_image, cv2.COLOR_RGB2YCrCb)
    if DEBUG: print("max car_sample: ", np.amax(car_sample))

    # Plot the examples
    fig = plt.figure()
    plt.subplot(241)
    plt.imshow(car_image)
    plt.title('Example Car Image')
    plt.subplot(245)
    plt.imshow(noncar_image)
    plt.title('Example Non-car Image')

    dt = DetectionTraining()
    tp = TrainingParameters()

    for channel in range(0, 3):
        car_hog_features, car_hog_images = dt.get_hog_features(car_sample[:,:,channel], orient=tp.orient,
                                                  pix_per_cell=tp.pix_per_cell, cell_per_block=tp.cell_per_block,
                                                  vis=True, feature_vec=True)
        noncar_hog_features, noncar_hog_images = dt.get_hog_features(noncar_sample[:,:,channel],  orient=tp.orient,
                                                  pix_per_cell=tp.pix_per_cell, cell_per_block=tp.cell_per_block,
                                                  vis=True, feature_vec=True)
        plt.subplot(240 + channel + 2)
        plt.imshow(car_hog_images)
        plt.title("ch-" + str(channel))
        plt.subplot(240 + channel + 6)
        plt.imshow(noncar_hog_images)
        plt.title("ch-" + str(channel))

        plt.savefig(filename="output_images/02-sample-hog-features.png")


def train_classifier():
    global car_features, noncar_features
    global svc, X_scaler

    print("Begin training classifier... Grab some coffee, this might take a while.")
    dt = DetectionTraining()
    tp = TrainingParameters()

    t = time.time()
    # Extract features
    car_features = dt.extract_features(cars, color_space=tp.color_space, spatial_size=tp.spatial_size,
                         hist_bins=tp.hist_bins, orient=tp.orient,
                         pix_per_cell=tp.pix_per_cell, cell_per_block=tp.cell_per_block, hog_channel=tp.hog_channel,
                         spatial_feat=True, hist_feat=True, hog_feat=True)
    noncar_features = dt.extract_features(noncars, color_space=tp.color_space, spatial_size=tp.spatial_size,
                         hist_bins=tp.hist_bins, orient=tp.orient,
                         pix_per_cell=tp.pix_per_cell, cell_per_block=tp.cell_per_block, hog_channel=tp.hog_channel,
                         spatial_feat=True, hist_feat=True, hog_feat=True)

    print("It took ", round(time.time()-t, 2), " seconds to compute features.")

    if DEBUG: print("In Train_classifer(), length of car_features: ", len(car_features))
    if DEBUG: print("In Train_classifer(), length of noncar_features: ", len(noncar_features))

    # Create an array stack of feature vectors
    X = np.vstack((car_features, noncar_features)).astype(np.float64)

    if DEBUG: print("In Train_classifer(), max X is ", np.amax(X))
    if DEBUG: print("In Train_classifer(), length of X: ", len(X))
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    if DEBUG: print("In Train_classifer(), max scaled X is ", np.amax(scaled_X))
    if DEBUG: print("In Train_classifer(), length of scaled_X: ", len(scaled_X))

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.1, random_state=rand_state)

    print("Using the following parameters: ")
    print("orientations: ", tp.orient)
    print("pixels per cell: ", tp.pix_per_cell)
    print("histogram bins: ", tp.hist_bins )
    print("spatial sampling size: ", tp.spatial_size)

    print("The length of feature vector in X_train is: ", len(X_train[0]))

    # Use a linear SVC
    svc = LinearSVC()
    t=time.time()
    svc.fit(X_train, y_train)
    print('Training SVC took ', round(time.time()-t, 2), 'seconds.')
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

def speedup_hog_on_test_images():
    out_images = []
    out_maps = []
    out_boxes = []
    out_titles = []

    sp = WindowSearchParameters()
    tp = TrainingParameters()
    dt = DetectionTraining()

    ystart = sp.y_start_stop[0]
    ystop = sp.y_start_stop[1]
    scale = sp.scale
    pix_per_cell = tp.pix_per_cell
    orient = tp.orient
    cell_per_block = tp.cell_per_block
    window = sp.window_size

    test_image_files = glob.glob("test_images/*.jpg")

    for f in test_image_files:
        img_boxes = []
        t = time.time()

        count = 0
        image = mpimg.imread(f)
        draw_image = np.copy(image)

        heatmap = np.zeros_like(image[:,:,0])
        img = image.astype(np.float32) / 255

        img_tosearch = img[ystart:ystop, :, :]
        ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        nxblocks = (ch1.shape[1] // pix_per_cell) -1
        nyblocks = (ch1.shape[0] // pix_per_cell) -1
        nfeat_per_block = orient*cell_per_block**2
        window = 64
        nblocks_per_window = (window // pix_per_cell) - 1
        cells_per_step = 2
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        hog1 = dt.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec = False)
        hog2 = dt.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec = False)
        hog3 = dt.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec = False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                count += 1
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step

                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                if DEBUG: print("hog_features shape: ", hog_features.shape)

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                if DEBUG: print("ctrans portion: ", ctrans_tosearch[ytop:ytop + window, xleft:xleft+window].shape)
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64, 64))
                if not DEBUG: print("subimg.shape: ", subimg.shape)

                spatial_features = dt.bin_spatial(subimg, size=tp.spatial_size)
                if not DEBUG: print("spatial_features shape: ", spatial_features.shape)

                hist_features = dt.color_hist(subimg, nbins=tp.hist_bins)
                if not DEBUG: print("hist_features shape: ", hist_features.shape)

                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                test_prediction = svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)

                    cv2.rectangle(draw_image, (xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart),(0,0,255))

                    img_boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw, ytop_draw+win_draw+ystart)))

                    heatmap[ytop_draw+ystart:ytop_draw+win_draw+ystart, xbox_left:xbox_left+win_draw] += 1

        print(round(time.time()-t, 2), "seconds to run, total windows = ", count)
        out_images.append(draw_image)
        out_titles.append(f[-10:])
        out_images.append(heatmap)
        out_titles.append(f[-10:])

        out_maps.append(heatmap)
        out_boxes.append(img_boxes)

    fig = plt.figure(figsize=(12,24))
    visualize(fig, 8, 2, out_images, out_titles)


def find_cars(image, scale):

    sp = WindowSearchParameters()
    tp = TrainingParameters()
    dt = DetectionTraining()

    ystart = sp.y_start_stop[0]
    ystop = sp.y_start_stop[1]
    scale = sp.scale
    pix_per_cell = tp.pix_per_cell
    orient = tp.orient
    cell_per_block = tp.cell_per_block
    window = sp.window_size

    img_boxes = []

    count = 0
    draw_image = np.copy(image)

    heatmap = np.zeros_like(image[:,:,0])
    img = image.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    nxblocks = (ch1.shape[1] // pix_per_cell) -1
    nyblocks = (ch1.shape[0] // pix_per_cell) -1
    nfeat_per_block = orient*cell_per_block**2
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    hog1 = dt.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec = False)
    hog2 = dt.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec = False)
    hog3 = dt.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec = False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            count += 1
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            if DEBUG: print("hog_features shape: ", hog_features.shape)

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            if DEBUG: print("ctrans portion: ", ctrans_tosearch[ytop:ytop + window, xleft:xleft+window].shape)
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64, 64))
            if DEBUG: print("subimg.shape: ", subimg.shape)

            spatial_features = dt.bin_spatial(subimg, size=tp.spatial_size)
            if DEBUG: print("spatial_features shape: ", spatial_features.shape)

            hist_features = dt.color_hist(subimg, nbins=tp.hist_bins)
            if DEBUG: print("hist_features shape: ", hist_features.shape)

            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)

                cv2.rectangle(draw_image, (xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart),(0,0,255))

                img_boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw, ytop_draw+win_draw+ystart)))

                heatmap[ytop_draw+ystart:ytop_draw+win_draw+ystart, xbox_left:xbox_left+win_draw] += 1

    return draw_image, heatmap

def find_cars_on_test_images():
    sp = WindowSearchParameters()
    s = SlidingWindowSearch()

    out_images = []
    out_titles = []
    out_maps = []
    ystart = sp.y_start_stop[0]
    ystop = sp.y_start_stop[1]
    scale = sp.scale

    test_image_files = glob.glob("test_images/*.jpg")

    for f in test_image_files:
        img = mpimg.imread(f)
        out_img, heat_map = find_cars(img, scale)
        labels = label(heat_map)
        draw_img = s.draw_labeled_bboxes(img, labels)
        out_images.append(draw_img)
        out_titles.append("image")
        out_images.append(heat_map)
        out_titles.append("heatmap")
    fig = plt.figure(figsize=(12,24))
    visualize(fig, 8, 2, out_images, out_titles)

def process_image2(img):  #using speedup hog process
    s = SlidingWindowSearch()
    sp = WindowSearchParameters()
    out_img, heat_map = find_cars(img, sp.scale)
    labels = label(heat_map)
    draw_img = s.draw_labeled_bboxes(np.copy(img), labels)
    return draw_img

def process_image3(img): #using average heatmap technique
    global heatmap_history

    s = SlidingWindowSearch()
    sp = WindowSearchParameters()
    out_img, heat_map = find_cars(img, sp.scale)
    #heat_map = s.apply_threshold(heat_map, 1)
    heatmap_history.append(heat_map)

    if len(heatmap_history) > HEATMAP_HISTORY_SIZE:
        heatmap_history.pop(0)

    #heatmap_with_history = sum(heatmap_history) * HEATMAP_HISTORICAL_RATIO + heat_map * (1- HEATMAP_HISTORICAL_RATIO)

    heatmap_with_history = sum(heatmap_history)
    heatmap_with_history = s.apply_threshold(heatmap_with_history, 5)

    labels = label(heatmap_with_history)
    draw_img = s.draw_labeled_bboxes(np.copy(img), labels)

    return draw_img


def search_and_classify(image):
    draw_image = np.copy(image)

    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    if DEBUG: print("in search and classify, image max :", np.amax(image))
    image = image.astype(np.float32)/255
    if DEBUG: print("in search and classify, image max :", np.amax(image))

    s = SlidingWindowSearch()
    sp = WindowSearchParameters()
    tp = TrainingParameters()
    windows = s.slide_window(image, x_start_stop=[None, None], y_start_stop=sp.y_start_stop,
                           xy_window=(sp.window_size, sp.window_size), xy_overlap=(0.5, 0.5))  #(96, 96)

    hot_windows = s.search_windows(image, windows, svc, X_scaler, color_space=tp.color_space,
                                 spatial_size=tp.spatial_size, hist_bins=tp.hist_bins,
                                 orient=tp.orient, pix_per_cell=tp.pix_per_cell,
                                 cell_per_block=tp.cell_per_block,
                                 hog_channel=tp.hog_channel, spatial_feat=tp.spatial_feat,
                                 hist_feat=tp.hist_feat, hog_feat=tp.hog_feat)

    window_img = s.draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    return hot_windows, window_img

def search_and_classify_test_images():
    test_image_files = glob.glob("test_images/*.jpg")

    count = 1
    for f in test_image_files:
        image = mpimg.imread(f)

        hot_windows, window_img = search_and_classify(image)
        fig = plt.figure()
        plt.imshow(window_img)
        if DEBUG: print("len(hot_windows)): ", len(hot_windows))
        plt.title(f)
        filename = "./output_images/test_images/" + str(count)
        count = count + 1
        plt.savefig(filename)
        plt.close()

def heatmap_search_test_images():
    s = SlidingWindowSearch()

    test_image_files = glob.glob("test_images/*.jpg")

    count = 0
    for f in test_image_files:
        image = mpimg.imread(f)
        draw_image = np.copy(image)
        heat = np.zeros_like(image[:, :, 0]).astype(np.float)

        hot_windows, window_img = search_and_classify(image)

        # Add heat to each hot_windows
        heat = s.add_heat(heat, hot_windows)

        # Apply threshold to help remove false positives
        #heat = s.apply_threshold(heat, 1)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        image_window = s.draw_labeled_bboxes(draw_image, labels)

        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(image_window)
        plt.title("Test Image" + str(count))
        plt.subplot(122)
        plt.imshow(heatmap)
        plt.title("Heatmap " + str(count))
        filename = "./output_images/heatmap_images/" + str(count)
        count = count + 1
        plt.savefig(filename)
        plt.close()

def process_image(image):
    s = SlidingWindowSearch()

    draw_image = np.copy(image)
    if DEBUG: print("max of image: ", np.amax(image))
    image = image.astype(np.float32)
    if DEBUG: print("max of image: ", np.amax(image))
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)

    hot_windows, window_img = search_and_classify(image)
    if DEBUG: print("len(hot_windows): ", len(hot_windows))

    # Add heat to each hot_windows
    heat = s.add_heat(heat, hot_windows)

    # Apply threshold to help remove false positives
    #heat = s.apply_threshold(heat, 1)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    labels = label(heatmap)

    result = s.draw_labeled_bboxes(draw_image, labels)

    return result



def run_video():
    clip = VideoFileClip('project_video.mp4')
    new_clip = clip.fl_image(process_image3)
    new_clip.write_videofile('project_video_w96_h15_t5.mp4', audio=False)

if __name__ == "__main__":
    retrieve_labeled_data()
    train_classifier()
    if WRITEUP: speedup_hog_on_test_images()
    find_cars_on_test_images()
    run_video()

'''
    # execute only if run as a script
    retrieve_labeled_data()
    if WRITEUP:
        plot_sample_images()
        speedup_hog_on_test_images()
    train_classifier()
    if WRITEUP:
        search_and_classify_test_images()
        heatmap_search_test_images()
    run_video()
'''