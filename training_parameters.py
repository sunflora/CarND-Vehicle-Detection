class TrainingParameters:
    def __init__(self):

        self.color_space = 'YCrCb'
        self.orient = 9
        self.pix_per_cell = 8
        self.cell_per_block = 2
        self.hog_channel = "ALL"
        self.spatial_size = (16, 16)  # Spatial binning dimensions
        self.hist_bins = 16  # Number of histogram bins
        self.spatial_feat = True  # Spatial features on or off
        self.hist_feat = True  # Histogram features on or off
        self.hog_feat = True  # HOG features on or off
